import time, json
from dataclasses import dataclass, asdict
from typing import Callable
from pathlib import Path
from pydantic import ValidationError
from parse import parse, ParseError
from prompts import SYSTEM_PROMPT, RETRY_HINT, BARE_PROMPT, SYSTEM_PROMPT_HASH
from client import real_llm_complete  # 真 API 客户端，网络层重试归它管
from contracts import LLMResult, sum_usage
from pricing import print_report  # 跑完自动出成本对比表，零依赖、不触发 API

PROBES_PATH = Path("probes.json")
OUT_PATH = Path("probes.jsonl")
LlmFn = Callable[[list[dict], float], LLMResult]  # 真函数 / 假函数同一个返回契约


@dataclass
class Result:
    llm_model: str
    run_tag: str
    probe_id: int
    repair_count: int
    final_ok: bool
    error_kind: str
    latency_ms: float
    output: dict | None
    error: str | None
    error_history: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_missing: int = 0  # 拿不到 usage 的轮次数，不假装成 0 token
    prompt_hash: str = ""  # 控制变量凭证：四档必须落同一个 hash


class RepaireExhausted(Exception):
    """带着失败记录抛出去：失败的那一行也必须落盘"""

    def __init__(self, result: Result):
        super().__init__(f"修复 {result.repair_count} 次后仍失败：{result.error_kind}")
        self.result = result


def _token_fields(usage_log: list) -> dict:
    """把每轮 usage 累加成 Result 的四个落盘字段。"""
    agg, missing = sum_usage(usage_log)
    return dict(
        prompt_tokens=agg.prompt_tokens,
        completion_tokens=agg.completion_tokens,
        total_tokens=agg.total_tokens,
        usage_missing=missing,
    )


def extract_one(
    probe: str,
    llm_complete: LlmFn,  # ← 接缝：真函数 / 假函数插同一个位置
    *,
    probe_id: int = 0,
    max_repair: int = 1,
    run_tag: str = "default",
    temperature: float = 0,  # T9 主实验=0；温度验证时传 0.3 / 0.8
):
    message = [  # 唯一的可变状态，循环外只建一次
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": probe},
    ]
    print("\n输入的内容是：" + probe)  # 每次输入都进行一次打印，看一下输入的内容
    started = time.perf_counter()
    last_kind, last_error = None, None
    error_history: list[str] = []
    usage_log = []  # 每轮一条，成功失败都记，用于「总体 token = Σ 每轮」
    llm_model = ""  # 循环外初始化：即使某轮 usage 为 None，model 也照样能落盘
    for attempt in range(max_repair + 1):  # +1：一次都不修也得先调一次
        res = llm_complete(message, temperature=temperature)
        usage_log.append(res.usage)
        raw = res.content
        if (
            attempt == 0
        ):  # model 在 LLMResult 上，不在 Usage 上；同一条样本以内以第一轮为准
            llm_model = res.model
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
            llm_model=llm_model,
            run_tag=run_tag,
            probe_id=probe_id,
            repair_count=attempt,  # 不维护计数器，直接读循环变量
            final_ok=True,
            error_kind=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            output=obj.model_dump(),
            error=None,
            error_history=error_history,
            prompt_hash=SYSTEM_PROMPT_HASH,
            **_token_fields(usage_log),
        )
    raise RepaireExhausted(
        Result(  # 出口 3：耗尽，绝不 return None
            llm_model=llm_model,
            run_tag=run_tag,
            probe_id=probe_id,
            repair_count=max_repair,
            final_ok=False,
            error_kind=last_kind,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            output=None,
            error=last_error,
            error_history=error_history,
            prompt_hash=SYSTEM_PROMPT_HASH,
            **_token_fields(usage_log),
        )
    )


def _write(result: Result):
    """落盘：jsonl"""
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")  # 中文别转义


def main(
    llm_complete: LlmFn,
    max_repair: int,
    run_tag: str,
    temperature: float = 0,
):
    probes = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
    grand_total_tokens = 0

    for i, probe in enumerate(probes):
        try:
            result = extract_one(
                probe,
                llm_complete,
                probe_id=i,
                max_repair=max_repair,
                run_tag=run_tag,
                temperature=temperature,
            )
        except RepaireExhausted as e:
            result = e.result
        grand_total_tokens += result.total_tokens
        _write(result)
        print(
            f"[{run_tag}][{result.llm_model}]{i+1}/{len(probes)} "
            f"ok={result.final_ok} repair={result.repair_count}"
        )
    print(f"[{run_tag}] 累计 tokens: {grand_total_tokens}")
    # 每档跑完自动出一次成本对比表：四档循环里第 4 次打印时即完整四档对比
    print_report(path=OUT_PATH)


if __name__ == "__main__":

    # ⚠ 单档冒烟行已关闭。它写的是 run_tag="deepseek/t9" 的短名 20 行，
    # 混进正式 80 行会让你误以为「四档循环没生效、还是只跑 DeepSeek」。
    # 需要单独冒烟时再打开这一行，跑完记得清空 probes.jsonl：
    # main(real_llm_complete, max_repair=0, run_tag="deepseek/t9")

    # 四档全量：run_tag 必须带模型全名，80 行才分得开。
    # 名实相符：7B / DeepSeek / Qwen72B / 混元 Hy4 各是独立常量，互不借名。
    from config import (
        SILICONFLOW_MODEL_QWEN7B,
        SILICONFLOW_MODEL_DEEPSEEK,
        SILICONFLOW_MODEL_QWEN,
        SILICONFLOW_MODEL_HUANYUAN_HY4,
    )
    from client import make_llm_complete

    for name in (
        SILICONFLOW_MODEL_QWEN7B,  # 便宜档 7B
        SILICONFLOW_MODEL_DEEPSEEK,  # 671B MoE
        SILICONFLOW_MODEL_QWEN,  # 72B 稠密
        SILICONFLOW_MODEL_HUANYUAN_HY4,  # 770B MoE 混元旗舰
    ):
        try:
            main(make_llm_complete(name), max_repair=0, run_tag=f"{name}/t9")
        except Exception as e:
            # 403 / 429 / 网络异常属 API 层，main 只接管 RepaireExhausted，会一路冒到这里。
            # 一档失败不拖垮整批：整档跳过、继续下一档，别让四档对比只剩一档。
            # 注意：若中途才失败，该档可能已写入部分行，重跑前请先清空 probes.jsonl。
            print(f"[{name}] ❌ 整档跳过：{type(e).__name__}: {e}")
            print_report(path=OUT_PATH)  # 坏档也出一次表，已跑的档照样看得到花费
