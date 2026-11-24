import json
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

from config import OUT_PATH, PROBES_PATH
from contracts import LLMResult, sum_usage
from parse import ParseError, parse
from pricing import print_report
from promps import RETRY_HINT, SYSTEM_PROMPT, SYSTEM_PROMPT_HASH
from temperature import TEMPERATURES, print_temperature_report

LlmFn = Callable[[list[dict], float], LLMResult]  # 真函数和假函数 同一个返回契约

# True即留档 JSONL文件
RESET_ON_RUN = True


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
    usage_missing: int = 0
    prompt_hash: str = ""
    temperature: float = 0.0


class RepairExhausted(Exception):
    """带着失败记录跑出去：失败的哪一行也必须落盘"""

    def __init__(self, result: Result):
        super().__init__(f"修复{result.repair_count}次后仍失败：{result.error_kind}")
        self.result = result


def _token_fields(usage_log: list):
    agg, missing = sum_usage(usage_log)
    return {
        "prompt_tokens": agg.prompt_tokens,
        "completion_tokens": agg.completion_tokens,
        "total_tokens": agg.total_tokens,
        "usage_missing": missing,
    }


def extract_one(
    probe: str,
    llm_complete: LlmFn,
    *,
    probe_id: int = 0,
    max_repair: int = 1,
    run_tag: str = "default",
    temperature: float = 0,
):
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": probe},
    ]

    print("\n输入的内容是：" + probe)  #  每次输入都进行一次打印，看一下输入的内容
    started = time.perf_counter()
    last_kind, last_error = None, None
    error_history: list[str] = []
    usage_log = []  # 每轮一条，成功失败都记，用于【总体 token = Σ 每轮】
    llm_model = ""

    for attempt in range(max_repair + 1):
        res = llm_complete(message, temperature=temperature)
        usage_log.append(res.usage)
        raw = res.content
        if attempt == 0:
            llm_model = res.model
        try:
            obj = parse(raw)
        except ParseError as e:
            last_kind, last_error = e.kind, e.message
            error_history.append(last_error)
            message += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"{RETRY_HINT}\n\n校验错误:\n{e.message}"},
            ]
            print("异常处理：" + e.message)  # 将异常情况展示

            continue
        return Result(
            llm_model=llm_model,
            run_tag=run_tag,
            probe_id=probe_id,
            repair_count=attempt,
            final_ok=True,
            error_kind=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            output=obj.model_dump(),
            error=None,
            error_history=error_history,
            prompt_hash=SYSTEM_PROMPT_HASH,
            temperature=temperature,
            **_token_fields(usage_log),
        )
    raise RepairExhausted(
        Result(
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
            temperature=temperature,
            **_token_fields(usage_log),
        )
    )


def _write(result: Result):
    """落盘：jsonl"""
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")  # 中文别转义


def _archive_and_reset():
    """整批重跑前留档 + 清空：留档行数校验通过才清空，绝不静默丢数据"""
    if not OUT_PATH.exists() or OUT_PATH.stat().st_size == 0:
        print(f"[{OUT_PATH}]已是空文件，无需留档")
        return
    n = sum(1 for l in OUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = OUT_PATH.with_name(f"probes_archive_{stamp}.jsonl")
    i = 1
    while archive.exists():
        archive = OUT_PATH.with_name(f"probes_archive_{stamp}_{i}.jsonl")
        i += 1
    shutil.copy(OUT_PATH, archive)
    m = sum(1 for l in archive.read_text(encoding="utf-8").splitlines() if l.strip())
    if m != n:
        raise RuntimeError(f"留档校验失败：源{n}行/档{m} 行，拒绝清空{OUT_PATH}")
    OUT_PATH.write_text("", encoding="utf-8")
    print(f"[{OUT_PATH}] 已留档{n} 行 -> {archive.name}，随后清空，本次从赶紧文件起跑")


def main(
    llm_complete: LlmFn,
    max_repair: int,
    run_tag: str,
    temperature: float = 0,
    report: bool = True,
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
        except RepairExhausted as e:
            result = e.result
        grand_total_tokens += result.total_tokens
        _write(result)
        print(
            f"[{run_tag}][{result.llm_model}]{i+1}/{len(probes)}"
            f"ok ={result.final_ok} repair = {result.repair_count}"
        )
    print(f"[{run_tag}] 累计 tokens： {grand_total_tokens}")
    if report:
        print_report(path=OUT_PATH)


if __name__ == "__main__":
    from client import make_llm_complete
    from config import (
        SILICONFLOW_MODEL_DEEPSEEK,
        SILICONFLOW_MODEL_GLM45AIR,
        SILICONFLOW_MODEL_QWEN7B,
        SILICONFLOW_MODEL_QWEN72B,
    )

    MODEL_ORDER = (
        SILICONFLOW_MODEL_QWEN7B,  # 免费档 7B
        SILICONFLOW_MODEL_GLM45AIR,  # 106B MoE，thinking 已统一关闭
        SILICONFLOW_MODEL_DEEPSEEK,  # 671B MoE
        SILICONFLOW_MODEL_QWEN72B,  # 72B 稠密
    )

    n_probes = len(json.loads(PROBES_PATH.read_text(encoding="utf-8")))

    if RESET_ON_RUN:
        _archive_and_reset()

    print(
        f"\n=== T9 温度扫描：{len(TEMPERATURES)} 温度 × {len(MODEL_ORDER)} 档 "
        f"× {n_probes} 条 = {len(TEMPERATURES) * len(MODEL_ORDER) * n_probes} 行 ==="
    )
    print("    控制变量：同一 prompt_hash、max_repair=0、统一关闭 thinking；")
    print("    自变量：模型档位 + temperature。成本表只认 t=0.0 的 /t9 行。")

    for t in TEMPERATURES:
        suffix = "/t9" if t == 0.0 else f"/temp{t}"
        for name in MODEL_ORDER:
            tag = f"{name}{suffix}"
            try:
                main(
                    make_llm_complete(name),
                    max_repair=0,
                    run_tag=tag,
                    temperature=t,
                    report=False,
                )
            except Exception as e:  # noqa: BLE001
                # 403 / 429 / 网络异常属 API 层，main 只接管 RepaireExhausted。
                # 一档失败不拖垮整批：整档跳过、继续下一档，最后照样出已有数据的结论。
                print(f"[{tag}] ❌ 整档跳过：{type(e).__name__}: {e}")

    # 统一出结论：只在这里打印一次，不再每个温度刷屏
    print("\n" + "#" * 100)
    print("# T9 统一结论  ① 成本表（temperature=0 / /t9 行）  ② 温度漂移分析")
    print("#" * 100)
    print_report(path=OUT_PATH)
    print_temperature_report(path=OUT_PATH)
