import time, json, shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable
from pathlib import Path
from pydantic import ValidationError
from parse import parse, ParseError
from prompts import SYSTEM_PROMPT, RETRY_HINT, BARE_PROMPT, SYSTEM_PROMPT_HASH
from client import real_llm_complete  # 真 API 客户端，网络层重试归它管
from contracts import LLMResult, sum_usage
from pricing import print_report  # 跑完自动出成本对比表，零依赖、不触发 API
from temperature import TEMPERATURES, print_temperature_report

PROBES_PATH = Path("probes.json")
OUT_PATH = Path("probes.jsonl")
LlmFn = Callable[[list[dict], float], LLMResult]  # 真函数 / 假函数同一个返回契约

# 整批重跑前是否自动「留档 + 清空」probes.jsonl。
# 原因：旧批次（尤其没有 temperature 字段、run_tag 只写 /t9 的那 80 行）混进新扫描
#   会无法自证温度，且按 (model, probe_id) 去重时和 temp=0 行同 key、互相覆盖。
# True = 直接 python pipeline.py 就从干净文件起跑，旧数据一律留档不丢；设 False 可自己管。
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
    usage_missing: int = 0  # 拿不到 usage 的轮次数，不假装成 0 token
    prompt_hash: str = ""  # 控制变量凭证：四档必须落同一个 hash
    temperature: float = 0.0  # 自变量之一：0 主实验 / 0.3 / 0.8 温度验证


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
    temperature: float = 0,  # 默认 = T9 主实验；温度验证由 main 显式传 0.3 / 0.8
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
            temperature=temperature,
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
            temperature=temperature,
            **_token_fields(usage_log),
        )
    )


def _write(result: Result):
    """落盘：jsonl"""
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")  # 中文别转义


def _archive_and_reset() -> None:
    """整批重跑前留档 + 清空：留档行数校验通过才清空，绝不静默丢数据。"""
    if not OUT_PATH.exists() or OUT_PATH.stat().st_size == 0:
        print(f"[{OUT_PATH}] 已是空文件，无需留档。")
        return
    n = sum(
        1 for l in OUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = OUT_PATH.with_name(f"probes_archive_{stamp}.jsonl")
    i = 1
    while archive.exists():  # 同秒重跑也不覆盖旧留档
        archive = OUT_PATH.with_name(f"probes_archive_{stamp}_{i}.jsonl")
        i += 1
    shutil.copy(OUT_PATH, archive)
    m = sum(
        1 for l in archive.read_text(encoding="utf-8").splitlines() if l.strip()
    )
    if m != n:
        raise RuntimeError(
            f"留档校验失败：源 {n} 行 / 档 {m} 行，拒绝清空 {OUT_PATH}"
        )
    OUT_PATH.write_text("", encoding="utf-8")
    print(f"[{OUT_PATH}] 已留档 {n} 行 → {archive.name}，随后清空，本次从干净文件起跑。")


def main(
    llm_complete: LlmFn,
    max_repair: int,
    run_tag: str,
    temperature: float = 0,
    report: bool = True,  # 三温扫描时置 False，最后统一出一次结论，不刷屏
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
    if report:  # 单跑一档时仍自动出成本表；三温扫描统一在 __main__ 末尾出
        print_report(path=OUT_PATH)


if __name__ == "__main__":
    # ── T9 温度扫描：一次跑完 0 / 0.3 / 0.8 × 四档，最后统一出结论 ──────────────
    # 三个 temperature 在同一进程内连续跑，消除「跨批次」变量（temp=0 已实证跨批次
    # 漂移 5~10pt）。每行 Result.temperature 显式落盘，温度不再只靠 run_tag 字符串猜。
    #
    # run_tag 规则（关键，决定哪批进成本表）：
    #   t=0.0 → "{name}/t9"      唯一进成本表的批次（pricing._latest_t9_rows 只认 /t9）
    #   t=0.3 → "{name}/temp0.3" 温度验证行，不进成本表
    #   t=0.8 → "{name}/temp0.8" 同上，且作为阳性对照（不漂 = 样本太简单）
    #
    # 控制变量：同一 SYSTEM_PROMPT（prompt_hash）、max_repair=0、四档统一关闭 thinking；
    # 自变量只有两个：模型档位 + temperature。四档按 input 单价升序，与 pricing.T9_MODELS 一致。
    #
    # ⚠ 换档 / 换温度必须整批重跑，不能只补一批：只补会造成 A/B 批次混淆，正确率不可比。
    # ⚠ Hy4 于 2026-11-20 剔除（推理模型跑短分类属任务错配，非「数据难看」），
    #   旧 80 行留档 probes_t9_20261120_with_hy4.jsonl，W5 报告引用它说明剔除理由。
    from config import (
        SILICONFLOW_MODEL_QWEN7B,
        SILICONFLOW_MODEL_GLM45AIR,
        SILICONFLOW_MODEL_DEEPSEEK,
        SILICONFLOW_MODEL_QWEN,
    )
    from client import make_llm_complete

    MODEL_ORDER = (
        SILICONFLOW_MODEL_QWEN7B,  # 免费档 7B
        SILICONFLOW_MODEL_GLM45AIR,  # 106B MoE，thinking 已统一关闭
        SILICONFLOW_MODEL_DEEPSEEK,  # 671B MoE
        SILICONFLOW_MODEL_QWEN,  # 72B 稠密
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
            except Exception as e:
                # 403 / 429 / 网络异常属 API 层，main 只接管 RepaireExhausted。
                # 一档失败不拖垮整批：整档跳过、继续下一档，最后照样出已有数据的结论。
                print(f"[{tag}] ❌ 整档跳过：{type(e).__name__}: {e}")

    # 统一出结论：只在这里打印一次，不再每个温度刷屏
    print("\n" + "#" * 100)
    print("# T9 统一结论  ① 成本表（temperature=0 / /t9 行）  ② 温度漂移分析")
    print("#" * 100)
    print_report(path=OUT_PATH)
    print_temperature_report(path=OUT_PATH)
