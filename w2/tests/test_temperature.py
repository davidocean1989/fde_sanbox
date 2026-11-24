"""温度漂移分析模块的单测：判据、去重、阳性对照、控制变量、结论输出。"""

from pathlib import Path

from pricing import T9_MODELS
from temperature import (
    BOUNDARY_PROBES,
    TEMPERATURES,
    analyze,
    collect,
    extract_temperature,
    print_temperature_report,
    resolve_model,
)

DS = "deepseek-ai/DeepSeek-V3"
GLM = "zai-org/GLM-4.5-Air"
NO_GOLD = Path("/nonexistent/gold.json")


def row(model, pid, t, topic, *, tag=None, with_temp=True, **kw):
    """造一行落盘记录。with_temp=False 模拟老批次（无 temperature 字段）。"""
    r = {
        "llm_model": model,
        "run_tag": (
            tag
            if tag is not None
            else (f"{model}/t9" if t == 0.0 else f"{model}/temp{t}")
        ),
        "probe_id": pid,
        "final_ok": True,
        "repair_count": 0,
        "error_history": [],
        "prompt_hash": "26c5fc13",
        "latency_ms": 1000.0,
        "prompt_tokens": 970,
        "completion_tokens": 41,
        "output": {"topic": topic, "confidence": 0.9, "reason": "mock"},
        **kw,
    }
    if with_temp:
        r["temperature"] = t
    return r


def test_extract_temperature_prefers_explicit_field():
    assert extract_temperature(row(DS, 0, 0.8, "hulu")) == 0.8
    assert extract_temperature({"temperature": 0.3, "run_tag": f"{DS}/t9"}) == 0.3


def test_extract_temperature_falls_back_to_run_tag():
    """老行没落 temperature 字段，但 run_tag 带 /temp0.3 → 仍能归属。"""
    assert extract_temperature(row(DS, 0, 0.3, "hulu", with_temp=False)) == 0.3


def test_old_t9_row_defaults_to_zero_and_is_lossy():
    """只写 /t9 又没字段的老行只能兜 0.0 —— 这正是必须重跑的原因。"""
    r = row(DS, 0, 0.3, "hulu", tag=f"{DS}/t9", with_temp=False)
    assert extract_temperature(r) == 0.0
    assert analyze([r], gold_path=NO_GOLD)["temp_field_missing"] == 1


def test_resolve_model_strips_both_tag_suffixes():
    assert resolve_model(row(DS, 0, 0.0, "hulu")) == DS
    assert resolve_model(row(DS, 0, 0.3, "hulu")) == DS
    assert resolve_model(row(DS, 0, 0.8, "hulu")) == DS
    # 老短名跑不动就回退 llm_model
    assert resolve_model(row(DS, 0, 0.0, "hulu", tag="deepseek/t9")) == DS
    assert resolve_model({"run_tag": "x/y", "llm_model": "not-in-T9"}) is None


def test_collect_dedup_keeps_last_like_pricing():
    """追加写导致同 (model, probe, temp) 多行时，保留最后一次。"""
    grid = collect(
        [
            row(DS, 5, 0.0, "public"),
            row(DS, 5, 0.0, "unknown"),  # 重跑覆盖
        ]
    )
    assert grid[(DS, 5)][0.0]["output"]["topic"] == "unknown"


def test_no_drift_when_all_temps_agree():
    rows = [row(DS, 5, t, "public") for t in TEMPERATURES]
    a = analyze(rows, gold_path=NO_GOLD)
    assert a["complete"] == 1
    assert a["drifts"] == []
    assert a["drift_03"] == 0 and a["drift_08"] == 0


def test_drift_detected_and_positive_control_fires():
    """0.8 判成 unknown → 阳性对照触发，明细 chain 与边界标记都对。"""
    rows = [
        row(DS, 5, 0.0, "public"),
        row(DS, 5, 0.3, "public"),
        row(DS, 5, 0.8, "unknown"),
    ]
    a = analyze(rows, gold_path=NO_GOLD)
    assert a["drift_08"] == 1
    assert a["drift_03"] == 0  # 低温段稳定
    assert len(a["drifts"]) == 1
    model, pid, topics = a["drifts"][0]
    assert (model, pid) == (DS, 5)
    assert [topics[t] for t in TEMPERATURES] == ["public", "public", "unknown"]
    assert a["drift_by_model"][DS] == 1


def test_low_temperature_drift_is_counted_separately():
    rows = [
        row(GLM, 16, 0.0, "rag"),
        row(GLM, 16, 0.3, "unknown"),  # 0→0.3 就漂
        row(GLM, 16, 0.8, "unknown"),
    ]
    a = analyze(rows, gold_path=NO_GOLD)
    assert a["drift_03"] == 1 and a["drift_08"] == 1


def test_boundary_probes_are_tracked_separately():
    rows = []
    for pid in BOUNDARY_PROBES:
        for t in TEMPERATURES:
            rows.append(row(DS, pid, t, "rag"))
    a = analyze(rows, gold_path=NO_GOLD)
    assert {pid for _m, pid, _c in a["boundary"]} == set(BOUNDARY_PROBES)
    assert a["complete"] == len(BOUNDARY_PROBES)


def test_control_variable_checks():
    rows = [row(DS, 0, t, "hulu") for t in TEMPERATURES]
    rows.append(row(GLM, 1, 0.0, "hulu", repair_count=1, final_ok=False))
    a = analyze(rows, gold_path=NO_GOLD)
    assert a["prompt_hashes"] == {"26c5fc13"}
    assert a["repair_events"] == 1
    assert a["final_ok_false"] == 1
    assert a["temp_field_missing"] == 0


def test_print_report_emits_conclusion_without_crashing(capsys):
    """三温数据齐 + 阳性对照触发 → 打印矩阵 / 明细 / 自动结论，全程不抛异常。"""
    rows = []
    for m in T9_MODELS:
        for pid in range(20):
            for t in TEMPERATURES:
                topic = "unknown" if (m == DS and pid == 5 and t == 0.8) else "hulu"
                rows.append(row(m, pid, t, topic))
    a = print_temperature_report(rows=rows, gold_path=NO_GOLD)
    out = capsys.readouterr().out
    assert "T9 温度漂移验证" in out
    assert "【自动结论】" in out
    assert "阳性对照" in out
    assert "主力建议" in out
    assert a["complete"] == 20 * len(T9_MODELS)
    assert a["drift_08"] == 1
    assert a["n_rows"] == 240


def test_print_report_warns_when_nothing_comparable(capsys):
    """只有一批温度时不能硬下结论，必须提示无法判定。"""
    rows = [row(DS, 0, 0.0, "hulu")]
    print_temperature_report(rows=rows, gold_path=NO_GOLD)
    out = capsys.readouterr().out
    assert "没有任何题目凑齐三档温度" in out


def test_dirty_old_batch_never_yields_a_routing_conclusion(capsys):
    """老批次无 temperature 字段被兜成 0.0：绝不能拿它下主力分工结论。"""
    rows = [
        row(m, pid, 0.0, "hulu", tag=f"{m}/t9", with_temp=False)
        for m in T9_MODELS
        for pid in range(20)
    ]
    a = print_temperature_report(rows=rows, gold_path=NO_GOLD)
    out = capsys.readouterr().out
    assert a["temp_field_missing"] == 80
    assert "本轮不下主力分工结论" in out
    assert "主力建议（" not in out
