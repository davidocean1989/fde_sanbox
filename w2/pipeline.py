import time, json
from dataclasses import dataclass, asdict
from typing import Callable
from pathlib import Path
from pydantic import ValidationError
from parse import parse, ParseError
from prompts import SYSTEM_PROMPT, RETRY_HINT, BARE_PROMPT
from client import real_llm_complete  # 真 API 客户端，网络层重试归它管

# TEMPERATURE = 0  # LLM04 三档：抽取/分类/结构化 = 0，写死不做旋钮
PROBES_PATH = Path("probes.json")
OUT_PATH = Path("probes.jsonl")
LlmFn = Callable[[list[dict], float], str]  # ?


@dataclass
class Result:
    run_tag: str
    probe_id: int
    repair_count: int
    final_ok: bool
    error_kind: str
    latency_ms: float
    output: dict | None
    error: str | None
    error_history: list[str]


class RepaireExhausted(Exception):
    """带着失败记录抛出去：失败的那一行也必须落盘"""

    def __init__(self, result: Result):
        super().__init__(f"修复 {result.repair_count} 次后仍失败：{result.error_kind}")
        self.result = result


def extract_one(
    probe: str,
    llm_complete: LlmFn,  # ← 接缝：真函数 / 假函数插同一个位置
    *,
    probe_id: int = 0,
    max_repair: int = 1,
    run_tag: str = "default",
):
    message = [  # 唯一的可变状态，循环外只建一次
        {"role": "system", "content": BARE_PROMPT},
        {"role": "user", "content": probe},
    ]
    print("\n输入的内容是：" + probe)  # 每次输入都进行一次打印，看一下输入的内容
    started = time.perf_counter()
    last_kind, last_error = None, None
    error_history: list[str] = []
    for attempt in range(max_repair + 1):  # +1：一次都不修也得先调一次
        raw = llm_complete(message, temperature=0)
        try:
            obj = parse(raw)
        except ParseError as e:
            last_kind, last_error = e.kind, e.message
            error_history.append(last_error)
            message += [  # 原地追加，messages 变长
                {"role": "assistant", "content": raw},  # 原始输出
                {"role": "user", "content": f"{RETRY_HINT}\n\n校验错误：\n{e.message}"},
            ]
            print("异常处理：" + e.message)  # 将异常情况展示

            continue

        return Result(  # 出口 1 / 2：一次过 / 修一次过
            run_tag=run_tag,
            probe_id=probe_id,
            repair_count=attempt,  # 不维护计数器，直接读循环变量
            final_ok=True,
            error_kind=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            output=obj.model_dump(),
            error=None,
            error_history=error_history,
        )
    raise RepaireExhausted(
        Result(  # 出口 3：耗尽，绝不 return None
            run_tag=run_tag,
            probe_id=probe_id,
            repair_count=max_repair,
            final_ok=False,
            error_kind=last_kind,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            output=None,
            error=last_error,
            error_history=error_history,
        )
    )


def _write(result: Result):
    """落盘：jsonl"""
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")  # 中文别转义


def main(llm_complete: LlmFn, max_repair: int, run_tag: str):
    probes = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
    for i, probe in enumerate(probes):
        try:
            result = extract_one(
                probe, llm_complete, probe_id=i, max_repair=max_repair, run_tag=run_tag
            )
        except RepaireExhausted as e:
            result = e.result
        _write(result)
        print(
            f"[{run_tag}]{i+1}/{len(probes)} ok={result.final_ok} repair={result.repair_count}"
        )


if __name__ == "__main__":

    main(real_llm_complete, max_repair=0, run_tag="baseline")  # A 组：关掉回喂
    main(real_llm_complete, max_repair=1, run_tag="with_repair")  # B 组：全量
