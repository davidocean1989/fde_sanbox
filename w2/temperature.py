# temperature.py -- T9 温度验证：漂移分析与统一结论
"""回答「同一批 20 条在 0 / 0.3 / 0.8 三档温度下，分类会不会漂」。

分层：pipeline 只负责跑和落盘实验事实（含 Result.temperature），本模块只做聚合与打印。
零业务依赖：只 import pricing（它本身零依赖）+ 标准库，import 不触发 API / 配置。
可单独复盘：python temperature.py —— 不重跑实验、不发请求。

判据与纪律：
- 漂移 = 同一 (model, probe_id) 在三档温度下 topic 不全相同。
- 0.8 是阳性对照：若 0.8 相对 0 一条都不漂，说明 20 条样本太简单，
  「正确率 / 漂移率」指标没有区分度，结论不能外推，需换难样本或扩样本。
- temperature=0 不保证跨批次复现（已实证独立运行差 1~2 条 = 5~10pt）。
  三温同一进程内连跑消除了「批次」变量，但 20 条里 1 条 = 5pt，仍不能把 5pt 当强结论。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# 复用 pricing 的「本轮测谁 / 短名 / temp=0 成本口径」，保持单一真相源。
# _SHORT 是 pricing 的模块私有映射，这里只在打印时用，不改它。
from pricing import T9_MODELS, _SHORT, summarize

TEMPERATURES: tuple[float, ...] = (0.0, 0.3, 0.8)
BOUNDARY_PROBES: tuple[int, ...] = (5, 16, 18)  # 词表边界题，优先盯漂移
GOLD_PATH = Path("gold.json")
_TEMP_IN_TAG = re.compile(r"/temp(\d+(?:\.\d+)?)$")


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 坏行跳过，别让一行脏数据毁掉整张表
    return out


def _load_gold(path: Path) -> dict[int, str]:
    if not Path(path).exists():
        return {}
    return {
        int(k): v for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()
    }


def extract_temperature(row: dict) -> float:
    """温度归属：优先显式字段；老行从 run_tag 的 /tempX 解析；再兜 0.0（主实验）。

    ⚠ 兜 0.0 是有损的：旧批次若 run_tag 只写 /t9、又没落 temperature 字段，
    就无法自证是 0 还是 0.3。analyze() 会统计这类行数并提示重跑前留档清空。
    """
    t = row.get("temperature")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    m = _TEMP_IN_TAG.search(row.get("run_tag") or "")
    return float(m.group(1)) if m else 0.0


def resolve_model(row: dict) -> str | None:
    """温度批次的模型归属：run_tag 去掉末尾 /t9 或 /tempX，回退 llm_model。"""
    tag = row.get("run_tag") or ""
    name = tag.rsplit("/", 1)[0] if "/" in tag else tag
    if name in T9_MODELS:
        return name
    model = row.get("llm_model") or ""
    return model if model in T9_MODELS else None


def _topic(row: dict) -> str | None:
    obj = row.get("output")
    return obj.get("topic") if isinstance(obj, dict) else None


def collect(rows: list[dict]) -> dict[tuple[str, int], dict[float, dict]]:
    """{(model, probe_id): {temp: row}}；每格保留最后一次（追加写去重，与 pricing 一致）。"""
    grid: dict[tuple[str, int], dict[float, dict]] = {}
    for row in rows:
        model, pid = resolve_model(row), row.get("probe_id")
        t = extract_temperature(row)
        if model is None or pid is None or t not in TEMPERATURES:
            continue
        grid.setdefault((model, pid), {})[t] = row
    return grid


def analyze(rows: list[dict], gold_path: Path = GOLD_PATH) -> dict:
    """纯计算，不打印不落盘。返回质量矩阵 / 漂移明细 / 边界题 / 控制变量。"""
    gold = _load_gold(Path(gold_path))
    grid = collect(rows)

    # 1) 温度 × 模型 质量矩阵
    grouped: dict[str, dict[float, list[dict]]] = {m: {} for m in T9_MODELS}
    for (model, _pid), cell in grid.items():
        for t, row in cell.items():
            grouped[model].setdefault(t, []).append(row)
    quality: dict[str, dict[float, dict]] = {}
    for model, by_t in grouped.items():
        quality[model] = {}
        for t, group in by_t.items():
            judged = correct = unknown = 0
            for r in group:
                topic = _topic(r)
                if topic == "unknown":
                    unknown += 1
                pid = r.get("probe_id")
                if pid in gold and topic is not None:
                    judged += 1
                    correct += topic == gold[pid]
            quality[model][t] = {
                "n": len(group),
                "accuracy": correct / judged if judged else None,
                "unknown_rate": unknown / len(group) if group else 0.0,
            }

    # 2) 漂移：同一 (model, probe) 三档 topic 不全相同
    drifts: list[tuple[str, int, dict[float, str | None]]] = []
    complete = 0
    for (model, pid), cell in grid.items():
        if all(t in cell for t in TEMPERATURES):
            complete += 1
        topics = {t: _topic(cell[t]) for t in TEMPERATURES if t in cell}
        if len(set(topics.values())) > 1:
            drifts.append((model, pid, topics))
    drifts.sort()
    drift_by_model = Counter(model for model, _pid, _topics in drifts)

    def changed(a: float, b: float) -> int:
        return sum(
            1
            for cell in grid.values()
            if a in cell and b in cell and _topic(cell[a]) != _topic(cell[b])
        )

    drift_03 = changed(0.0, 0.3)
    drift_08 = changed(0.0, 0.8)  # 阳性对照

    # 3) 边界题三温对照
    boundary = [
        (model, pid, {t: _topic(cell[t]) for t in TEMPERATURES if t in cell})
        for (model, pid), cell in grid.items()
        if pid in BOUNDARY_PROBES
    ]
    boundary.sort()

    # 4) 控制变量校验
    prompt_hashes = {r.get("prompt_hash") for r in rows if r.get("prompt_hash")}
    repair_events = sum(
        1 for r in rows if r.get("repair_count", 0) > 0 or r.get("error_history")
    )
    final_ok_false = sum(1 for r in rows if not r.get("final_ok", True))
    temp_field_missing = sum(1 for r in rows if "temperature" not in r)

    return {
        "n_rows": len(rows),
        "n_cells": len(grid),
        "complete": complete,
        "quality": quality,
        "drifts": drifts,
        "drift_by_model": drift_by_model,
        "drift_03": drift_03,
        "drift_08": drift_08,
        "boundary": boundary,
        "prompt_hashes": prompt_hashes,
        "repair_events": repair_events,
        "final_ok_false": final_ok_false,
        "temp_field_missing": temp_field_missing,
    }


def print_temperature_report(
    rows: list[dict] | None = None,
    path: Path = Path("probes.jsonl"),
    gold_path: Path = GOLD_PATH,
) -> dict:
    """打印温度漂移报告 + 自动结论。跑完三温后由 pipeline 统一调用一次。"""
    rows = _read_jsonl(Path(path)) if rows is None else rows
    a = analyze(rows, gold_path)
    line = "=" * 100
    print("\n" + line)
    print(
        f"T9 温度漂移验证 | 温度 {list(TEMPERATURES)} | "
        f"同一次运行内对比（批次变量已消除）"
    )
    print(line)

    # 表 A：温度 × 模型 质量矩阵
    print(f"{'model':<16}" + "".join(f"{'t=' + str(t):^16}" for t in TEMPERATURES))
    print(f"{'':<16}" + "".join(f"{'acc':>8}{'unknown':>8}" for _ in TEMPERATURES))
    print("-" * 100)
    for model in T9_MODELS:
        cells = ""
        for t in TEMPERATURES:
            d = a["quality"][model].get(t)
            if not d:
                cells += f"{'未跑':>8}{'—':>8}"
            else:
                acc = "n/a" if d["accuracy"] is None else f"{d['accuracy']:.0%}"
                cells += f"{acc:>8}{d['unknown_rate']:>8.0%}"
        print(f"{_SHORT[model]:<16}{cells}")

    # 表 B：漂移总览
    print("-" * 100)
    print(
        f"漂移总览：可三温对比的题目 {a['complete']} 个，"
        f"发生漂移 {len(a['drifts'])} 个"
    )
    print(
        f"  0 → 0.3 变化 {a['drift_03']} 条 | "
        f"0 → 0.8 变化 {a['drift_08']} 条（阳性对照）"
    )
    print(
        "  分档漂移："
        + " | ".join(f"{_SHORT[m]} {a['drift_by_model'].get(m, 0)}" for m in T9_MODELS)
    )

    # 表 C：漂移明细
    print("-" * 100)
    if a["drifts"]:
        print("漂移明细（probe | 0 → 0.3 → 0.8）：")
        for model, pid, topics in a["drifts"]:
            chain = " → ".join(str(topics.get(t, "—")) for t in TEMPERATURES)
            mark = "  ⭐边界题" if pid in BOUNDARY_PROBES else ""
            print(f"  {_SHORT[model]:<16} probe {pid:>2} : {chain}{mark}")
    else:
        print("漂移明细：三档温度全一致，无漂移。")

    # 表 D：边界题
    print("-" * 100)
    print(f"边界题 {list(BOUNDARY_PROBES)} 三温对照：")
    if a["boundary"]:
        for model, pid, topics in a["boundary"]:
            chain = " → ".join(str(topics.get(t, "—")) for t in TEMPERATURES)
            print(f"  {_SHORT[model]:<16} probe {pid:>2} : {chain}")
    else:
        print("  （无边界题数据）")

    # 控制变量
    print("-" * 100)
    print(
        f"控制变量：prompt_hash {sorted(a['prompt_hashes']) or '（无）'} | "
        f"RepairEvent {a['repair_events']} | final_ok=False {a['final_ok_false']} | "
        f"落盘 {a['n_rows']} 行"
    )
    if a["temp_field_missing"]:
        print(
            f"⚠ {a['temp_field_missing']} 行没有 temperature 字段（按 run_tag / 0.0 兜底）"
            f"—— 老批次无法自证温度，整批重跑前请留档并清空 probes.jsonl。"
        )

    # 自动结论
    print("-" * 100)
    print("【自动结论】")
    if a["complete"] == 0:
        print("  ⚠ 没有任何题目凑齐三档温度，无法判定漂移。")
        print(
            "     检查三档是否都跑完、run_tag 是否带温度后缀、temperature 字段是否落盘。"
        )
    elif a["drift_08"] == 0:
        print("  ⚠ 阳性对照未触发：0.8 相对 0 一条都没漂。20 条样本对温度不敏感，")
        print("     「正确率 / 漂移率」指标没有区分度，不能拿 5pt 差距外推；")
        print("     W5 前请换难样本或扩样本，否则温度验证这一关卡等于没测。")
    else:
        print(
            f"  ✅ 0.8 相对 0 漂了 {a['drift_08']} 条，温度确实影响分类；"
            f"0→0.3 漂 {a['drift_03']} 条"
            f"（低温段{'稳定' if a['drift_03'] == 0 else '已有漂移'}）。"
        )
        stable = sorted(T9_MODELS, key=lambda m: a["drift_by_model"].get(m, 0))
        print(
            f"     最稳：{_SHORT[stable[0]]}（{a['drift_by_model'].get(stable[0], 0)} 条）| "
            f"最不稳：{_SHORT[stable[-1]]}（{a['drift_by_model'].get(stable[-1], 0)} 条）。"
        )

    # 主力建议：复用 pricing 的 temp=0 成本口径（/t9 行），不重复实现计价。
    # ⚠ 数据不干净（没凑齐三温 / 含无 temperature 字段的老行）时不下分工结论，
    #   否则会把「被兜成 0.0 的 0.3 批次」当成主实验，等于拿脏数据定分工。
    s = summarize(rows, gold_path=gold_path)
    priced = [s[m]["year_cost"] for m in s if s[m]["year_cost"] is not None]
    data_clean = a["complete"] > 0 and a["temp_field_missing"] == 0
    if s and priced and data_clean:
        ranked = sorted(
            (m for m in T9_MODELS if m in s),
            key=lambda m: (
                -(s[m]["accuracy"] if s[m]["accuracy"] is not None else -1),
                s[m]["year_cost"] if s[m]["year_cost"] is not None else float("inf"),
            ),
        )
        best = ranked[0]
        b = s[best]
        acc_s = "n/a" if b["accuracy"] is None else f"{b['accuracy']:.0%}"
        best_year = b["year_cost"] if b["year_cost"] is not None else 0.0
        print(
            f"  主力建议（temperature=0 的 acc + 年成本）：{_SHORT[best]} "
            f"acc {acc_s} / 年 ¥{best_year:.2f}；"
            f"相对最贵在测档年省 ¥{max(priced) - best_year:.2f}。"
        )
    else:
        print("  ⚠ 数据不足（未凑齐三温 / 含无温度字段的老行），本轮不下主力分工结论；")
        print("     整批重跑出干净 240 行后再定。")

    print("  ⚠ 20 条样本 1 条 = 5pt；temperature=0 已实证跨批次漂移 5~10pt，")
    print("     三温同批跑只消除批次变量，5pt 差距仍可能是噪声，勿写进强结论。")
    print("  ⚠ T9 是强 prompt 下的乐观下界：生产需乘 (1+E[修复次数])，回喂见 T10；")
    print("     格式一次通过率是控制变量校验列，不参与排名。")
    print(line + "\n")
    return a


if __name__ == "__main__":
    # 不重跑实验、不发 API，直接基于现有 probes.jsonl 复盘：python temperature.py
    print_temperature_report()
