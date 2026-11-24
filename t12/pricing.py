# pricing.py -- 价格表与成本计算
"""T9 成本口径：单价单位「元 / 1K tokens」（与 config.py 各模型注释一致）。

职责边界：
- 只回答「这些 token 值多少钱」，不修改 probes.jsonl 的任何一行。
- 价格是会变的「分析时维度」，绝不写进落盘记录；落盘只记 token 这个实验事实。
  好处：查到新单价后直接改本表 + python pricing.py 重算，不用重跑任何一次 API。
- 单价未核对的模型在价格表里置 None：报表显示 n/a，绝不静默按 0 元计入成本表。
  注意区分「核对过就是 0 元」(Qwen2.5-7B Free) 与「价格未知」(None)——前者合法计价为 0。
- 单位固定 /1000，不是 /1e6 —— 两者差 1000 倍，别混。
- 本模块零依赖：不 import config / client / pipeline，import 它不会触发 API 配置。
"""

import json
import math
from pathlib import Path

# 单元：元 / 1K tokens（人民币）。 与 config.py 里个模型注释的单价对齐
PRICE_UNIT = "CNY_PER_1K_TOKENS"
PRICE_DIVISOR = 1000.0

# 本次 T9 的四档（有序 = 报表行序 = 按 input 单价升序）。
# 与 MODEL_PRICING 解耦：价格表是「已知单价的全集」，这里只声明「这轮测谁」。
# 换档只改这里 + config.py 常量 + pipeline.py 循环，价格字典不必跟着动。
T9_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-7B-Instruct",  # 免费档（官方 Free，0 元）
    "zai-org/GLM-4.5-Air",  # 低价 MoE，必须关闭 thinking
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-72B-Instruct",
)

# 模型全名 (config 常量) : (input 单价, output 单价)；None = 单价未核对
MODEL_PRICING: dict[str, tuple[float, float] | None] = {
    # 官方标记 (Free)，2026-11-20 控制台核对：输入 / 输出均 ¥0.000000 / K Tokens。
    # ⚠ 0 元 ≠格，与「不知道价格而置 None」是两回事。
    #   免费档的真实代价在质量侧（u 缺失：这是核对过的真实价nknown 40% 要转人工/二级处理），不进 T9 的 API 账单。
    "Qwen/Qwen2.5-7B-Instruct": (0.000000, 0.000000),
    "zai-org/GLM-4.5-Air": (0.001000, 0.006000),  # 106B MoE / 128K
    # 口径统一：各档 prompt_tokens_details.cached_tokens 一律不拆分、按全价计。
    # GLM 冒烟实测命中 320 cached tokens，官方价目表无「缓存命中」行，故报表是保守上界；
    # 四档都按全价算，横向对比仍成立（真实账单可能略低于本表）。
    "deepseek-ai/DeepSeek-V3": (0.002000, 0.008000),  # 671B MoE / 激活 37B
    "Qwen/Qwen2.5-72B-Instruct": (0.004130, 0.004130),  # 72B 稠密
    # 2026-11-20 已从 T9 剔除（推理模型跑短分类属任务错配），保留在价格表里
    # 只为让旧留档 probes_t9_20261120_with_hy4.jsonl 仍可重算审计，报表不显示这一行。
    "tencent/Hy4-preview": (0.006000, 0.018000),
}

_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "zai-org/GLM-4.5-Air": "GLM-4.5-Air",
    "deepseek-ai/DeepSeek-V3": "DeepSeek-V3",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "tencent/Hy4-preview": "Hy4-preview",
}

DAILY_CALLS = 1000  # 日均调用次数，用于折算月/年成本
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365
T9_TAG_SUFFIX = "/t9"


def cost(prompt_tokens: int, completion_tokens: int, price: tuple[float, float]):
    p_in, p_out = price
    return (prompt_tokens * p_in + completion_tokens * p_out) / PRICE_DIVISOR


def resolve_model(row: dict):
    """从一行落盘记录解析出价格表里的模型名。

    优先用 run_tag（实验者自己指定的请求名，形如 '{name}/t9'，可控）；
    解析不出时回退到 llm_model（服务端 resp.model，可能带版本后缀）。
    都认不出来返回 None —— 宁可跳过，也不静默按 0 元算。
    """
    tag = row.get("run_tag") or ""
    if tag.endswith(T9_TAG_SUFFIX):
        name = tag[: -len(T9_TAG_SUFFIX)]
        if name in MODEL_PRICING:
            return name
    model = row.get("llm_model") or ""
    return model if model in MODEL_PRICING else None


def row_cost(row: dict):
    """一行记录的成本。模型认不出 / 单价未核对 / token 字段缺失 → None（不假装成 0）。"""
    price = MODEL_PRICING.get(resolve_model(row) or "")
    if price is None:
        return None
    p, c = row.get("prompt_tokens"), row.get("completion_tokens")
    if p is None or c is None:  # 兼容历史行：老行可能没有这两个字段
        return None
    return cost(p, c, price)


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_gold(path: Path):
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def _latest_t9_rows(rows: list[dict]):
    """按 (模型, probe_id) 去重，保留最后出现的一行。

    probes.jsonl 是追加写：重跑同一档、或旧冒烟行混在文件里时，
    同一 probe 只算最后一次，保证每档永远是干净的 20 行。
    """
    latest: dict[tuple[str, int], dict] = {}
    for row in rows:
        if not (row.get("run_tag") or "").endswith(T9_TAG_SUFFIX):
            continue
        model = resolve_model(row)
        pid = row.get("probe_id")
        if model is None or pid is None:
            continue
        latest[(model, pid)] = row
    return list(latest.values())


def summarize(rows: list[dict], gold_path: Path = Path("gold.json")):
    gold = _load_gold(Path(gold_path))
    by_model: dict[str, list[dict]] = {}
    for row in _latest_t9_rows(rows):
        by_model.setdefault(resolve_model(row), []).append(row)

    out = {}
    for model, group in by_model.items():
        costs = [c for c in (row_cost(r) for r in group) if c is not None]
        if costs:
            avg = sum(costs) / len(costs)
            total = sum(costs)
            month = avg * DAILY_CALLS * DAYS_PER_MONTH
            year = avg * DAILY_CALLS * DAYS_PER_YEAR
        else:
            avg = total = month = year = None

        judged = correct = unknown = first_pass = 0
        for r in group:
            if r.get("final_ok") and r.get("repair_count", 0) == 0:
                first_pass += 1
            obj = r.get("output")
            topic = obj.get("topic") if isinstance(obj, dict) else None
            if topic == "unknown":
                unknown += 1
            if r.get("probe_id") in gold and topic is not None:
                judged += 1
                correct += topic == gold[r["probe_id"]]
        lat = sorted(r["latency_ms"] for r in group if r.get("latency_ms") is not None)

        out[model] = {
            "n": len(group),
            "priced_n": len(costs),
            "avg_cost": avg,
            "total_cost": total,
            "month_cost": month,
            "year_cost": year,
            "accuracy": correct / judged if judged else None,
            "unknown_rate": unknown / len(group) if group else 0.0,
            "first_pass_rate": first_pass / len(group) if group else 0.0,
            "lat_mean_ms": sum(lat) / len(lat) if lat else 0.0,
            "lat_p95_ms": lat[math.ceil(0.95 * len(lat)) - 1] if lat else 0.0,
        }
    return out


def _fmt(v, spec: str):
    if v is not None:
        return format(v, spec)
    width = int(spec.lstrip(">").split(".")[0])
    return "n/a".rjust(width)


def print_report(rows: list[dict] | None = None, path: Path = Path("probes.jsonl")):
    """打印四档成本对比表。固定 T9_MODELS 四行，没跑的档显示「未跑」。"""
    s = summarize(_read_jsonl(Path(path)) if rows is None else rows)
    print("\n" + "=" * 96)
    print(f"T9 四档成本对比 | 单位 {PRICE_UNIT} | 月 / 年按日均 {DAILY_CALLS} 次折算")
    print("=" * 96)
    print(
        f"{'model':<16}{'n':>4}{'avg-CNY':>12}{'total-CNY':>12}{'month-CNY':>11}"
        f"{'year-CNY':>10}{'acc':>7}{'unknown':>9}{'1st-pass':>10}{'lat-ms':>9}{'p95-ms':>9}"
    )
    print("-" * 96)

    for model in T9_MODELS:
        short = _SHORT[model]
        if model not in s:
            print(f"{short:<16}{'未跑':>4}")
            continue
        d = s[model]
        acc = "n/a" if d["accuracy"] is None else f"{d['accuracy']:.0%}"
        print(
            f"{short:<16}{d['n']:>4}"
            f"{_fmt(d['avg_cost'], '>12.6f')}{_fmt(d['total_cost'], '>12.6f')}"
            f"{_fmt(d['month_cost'], '>11.2f')}{_fmt(d['year_cost'], '>10.2f')}"
            f"{acc:>7}{d['unknown_rate']:>9.0%}{d['first_pass_rate']:>10.0%}"
            f"{d['lat_mean_ms']:>9.0f}{d['lat_p95_ms']:>9.0f}"
        )
        unpriced = [m for m in T9_MODELS if MODEL_PRICING.get(m) is None]
        print("-" * 96)
        if unpriced:
            print(f"⚠ 单价未核对，成本列 n/a：{', '.join(unpriced)}")
            print(
                "  到 cloud.siliconflow.cn/models 查价后填入 MODEL_PRICING；价格是分析时维度，"
            )
            print("  改完直接 python pricing.py 重算即可，不用重跑实验。")
        print("口径：单条均价 = (Σprompt×P_in + Σcompletion×P_out)/1000；")
        print("      正确率对照 gold.json；P95 取排序后 ceil(0.95×N)-1，不是均值。")
        print(
            "⚠ 四档统一关闭推理模式（enable_thinking=False，见 client.py），模型档位仍是唯一自变量。"
        )
        print("⚠ T9 是强 prompt 下的乐观下界：生产需乘 (1+E[修复次数])，回喂见 T10。")
        print(
            "⚠ acc / unknown / 1st-pass 属质量表：格式一次通过率是控制变量校验列，不参与排名。"
        )
        print("=" * 96 + "\n")
        return s
