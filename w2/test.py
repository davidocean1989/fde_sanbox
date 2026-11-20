from openai import OpenAI
from pathlib import Path
from statistics import median, mean
from parse import _strip_fence
import math, time, json

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL_DEEPSEEK,
    SILICONFLOW_MODEL_GLM45AIR,
    SILICONFLOW_MODEL_QWEN,
    SYSTEM_PROMPT,
)

# client 建一次就够：60 次调用复用同一条连接，省掉 59 次 TCP+TLS 握手
_client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)

T9_EXTRA_BODY = {"enable_thinking": False}

MODELS = [
    SILICONFLOW_MODEL_DEEPSEEK,
    SILICONFLOW_MODEL_GLM45AIR,
    SILICONFLOW_MODEL_QWEN,
]

SHORT = {
    SILICONFLOW_MODEL_DEEPSEEK: "DeepSeek-V3",
    SILICONFLOW_MODEL_GLM45AIR: "GLM-4.5-Air",
    SILICONFLOW_MODEL_QWEN: "Qwen2.5-72B",
}

# 单价：元 / 百万 token，格式为 (输入单价, 输出单价)
PRICE = {
    SILICONFLOW_MODEL_DEEPSEEK: (2.0, 8.0),  # 官方公告 2025-02-09 恢复原价
    SILICONFLOW_MODEL_GLM45AIR: (1.0, 6.0),  # 官方控制台 2026-11-20 核对：0.001/0.006 元每 K
    SILICONFLOW_MODEL_QWEN: (
        4.13,
        4.13,
    ),  # 官方控制台截图：¥0.004130/K Tokens，输入输出同价
}


def _fn(model: str, probe: str, probe_id: int) -> dict:
    """跑一次流式调用，返回一条结构化记录；不在内部 print。"""
    message = [  # 唯一的可变状态，循环外只建一次
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": probe},
    ]
    t0 = time.perf_counter()  # 记录 start 时间
    resp = _client.chat.completions.create(
        model=model,
        messages=message,
        temperature=0,
        max_tokens=5000,
        extra_body=T9_EXTRA_BODY,
        stream=True,
        stream_options={"include_usage": True},  # 关键：让最后一帧带上 usage 统计
    )
    full_text = ""

    t1 = None
    usage = None  # usage 汇总帧才更新，先占位
    for event in resp:

        # 区分依据必须是 choices 是否为空，不是 usage 是否为 None：
        # SiliconFlow 每个内容帧都带 usage（首帧 completion_tokens=0），
        # 只有 choices 为空的那一帧才是最终 usage 汇总。
        if not event.choices:
            if event.usage is not None:
                usage = event.usage
            continue
        delta_content = event.choices[0].delta.content
        if not delta_content:  # 判断是否为 None、""、0、[]、{}、False 时都会进来
            continue
        if t1 is None:
            t1 = time.perf_counter()  # 第一个 token 的时候记时间
        full_text += delta_content
    end = time.perf_counter()  # 结束时候的时间

    # 兜底：让失败原因明确，而不是在下面抛 NoneType 的 TypeError
    if t1 is None:
        raise RuntimeError("没有收到任何内容帧，检查 model 名称和 stream_options")
    if usage is None:
        raise RuntimeError("没有收到 usage 汇总帧，检查该模型是否支持 include_usage")

    # 防御：模型偶尔会加围栏或解释文字，_strip_fence 抽净再解析。
    # 20 条长批次里只要一条裸 json.loads 抛异常，整轮就白跑。
    obj = json.loads(_strip_fence(full_text))

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    pin, pout = PRICE[model]
    total_cost = prompt_tokens / 1_000_000 * pin + completion_tokens / 1_000_000 * pout

    return {
        "model": model,
        "probe_id": probe_id,
        "ttft_ms": round((t1 - t0) * 1000, 1),
        "tzcd_ms": round((end - t1) * 1000, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost": total_cost,
        "topic": obj.get("topic"),
    }


def _p95(values: list[float]) -> float:
    """口径与 pricing.py:179 一致：排序后 ceil(0.95*n)-1，不是均值。"""
    s = sorted(values)
    return s[math.ceil(0.95 * len(s)) - 1]


def _report(records: list[dict]):
    """把 60 条记录聚合成一张表 + 三个排名。"""
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    print("\n" + "=" * 104)
    print("流式 20-probe 测量 | 控制变量：同 probes / 同 prompt / temp=0 / thinking=OFF")
    print("=" * 104)
    print(
        f"{'model':<14}{'n':>3}{'TTFT中位':>10}{'TTFT均值':>10}{'TTFT-P95':>10}"
        f"{'吐字中位':>10}{'长尾>1.5s':>10}{'token合计':>12}{'成本合计':>12}"
    )
    print("-" * 104)

    stats = {}
    for model in MODELS:
        group = by_model.get(model, [])
        if not group:
            print(f"{SHORT[model]:<14}{'未跑':>3}")
            continue
        ttfts = [r["ttft_ms"] for r in group]
        tzcd = [r["tzcd_ms"] for r in group]
        psum = sum(r["prompt_tokens"] for r in group)
        csum = sum(r["completion_tokens"] for r in group)
        pin, pout = PRICE[model]
        cost_sum = psum / 1_000_000 * pin + csum / 1_000_000 * pout
        d = {
            "n": len(group),
            "median": median(ttfts),
            "mean": mean(ttfts),
            "p95": _p95(ttfts),
            "tzcd_median": median(tzcd),
            "long_tail": sum(1 for t in ttfts if t > 1500),
            "tokens": f"{psum}/{csum}",
            "cost": cost_sum,
        }
        stats[model] = d
        print(
            f"{SHORT[model]:<14}{d['n']:>3}{d['median']:>10.1f}{d['mean']:>10.1f}"
            f"{d['p95']:>10.1f}{d['tzcd_median']:>10.1f}{d['long_tail']:>10}"
            f"{d['tokens']:>12}{d['cost']:>12.6f}"
        )

    print("-" * 104)
    rank_median = sorted(stats, key=lambda m: stats[m]["median"])
    rank_mean = sorted(stats, key=lambda m: stats[m]["mean"])
    rank_cost = sorted(stats, key=lambda m: stats[m]["cost"])
    print("体感榜（TTFT 中位数）:", " < ".join(SHORT[m] for m in rank_median))
    print("体感榜（TTFT 均值）  :", " < ".join(SHORT[m] for m in rank_mean))
    print("成本榜（token 累加）  :", " < ".join(SHORT[m] for m in rank_cost))
    print("体感 vs 成本一致？     :", "一致" if rank_median == rank_cost else "不一致")
    print(
        "中位数榜 vs 均值榜？   :",
        "一致" if rank_median == rank_mean else "不一致 ← 抖动制造的假象",
    )
    print("=" * 104 + "\n")


PROBES_PATH = Path("probes.json")
probes = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
records: list[dict] = []
for probe_id, probe in enumerate(probes):
    for model in MODELS:
        try:
            r = _fn(model, probe, probe_id)
            records.append(r)
        except Exception as e:
            # 单条失败不拖垮整批，但要留痕，免得 n 少了一条还以为跑全了
            print(f"[跳过] probe={probe_id} model={SHORT[model]}: {type(e).__name__}: {e}")
        print(f"[{probe_id + 1}/{len(probes)}] {SHORT[model]} done")

_report(records)
