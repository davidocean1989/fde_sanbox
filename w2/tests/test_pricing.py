import pytest
from pathlib import Path
from pricing import (
    MODEL_PRICING,
    PRICE_UNIT,
    PRICE_DIVISOR,
    cost,
    row_cost,
    resolve_model,
    summarize,
)


def test_cost_matches_real_probe0():
    """用 DeepSeek probe 0 的真实 token 手算对账：970×0.002 + 41×0.008 = 0.002268 元。"""
    price = MODEL_PRICING["deepseek-ai/DeepSeek-V3"]
    assert cost(970, 41, price) == pytest.approx(0.002268, abs=1e-9)


def test_unit_is_per_1k_not_per_million():
    """防单位回归：/1000 与 /1e6 差整整 1000 倍，第一轮口径就差点踩这个坑。"""
    assert PRICE_UNIT == "CNY_PER_1K_TOKENS"
    assert PRICE_DIVISOR == 1000.0
    price = MODEL_PRICING["deepseek-ai/DeepSeek-V3"]
    assert cost(970, 41, price) == pytest.approx(
        (970 * 0.002 + 41 * 0.008) / 1e6 * 1000, abs=1e-9
    )


def test_unknown_model_never_silently_priced_zero():
    """认不出的模型必须返回 None，不能静默按 0 元混进成本表。"""
    row = {
        "llm_model": "some/model-not-in-table",
        "run_tag": "some/t9",
        "prompt_tokens": 970,
        "completion_tokens": 41,
    }
    assert resolve_model(row) is None
    assert row_cost(row) is None


def test_missing_token_field_returns_none():
    """历史行可能没有 token 字段，用 .get 兜底，别当 0。"""
    row = {"llm_model": "deepseek-ai/DeepSeek-V3", "run_tag": "deepseek/t9"}
    assert row_cost(row) is None


def test_resolve_model_prefers_run_tag_but_falls_back():
    """四档循环 run_tag 是 '{全名}/t9'；旧冒烟行是短名，靠 llm_model 回退。"""
    new = {"llm_model": "resp-name-with-suffix", "run_tag": "deepseek-ai/DeepSeek-V3/t9"}
    old = {"llm_model": "deepseek-ai/DeepSeek-V3", "run_tag": "deepseek/t9"}
    assert resolve_model(new) == "deepseek-ai/DeepSeek-V3"
    assert resolve_model(old) == "deepseek-ai/DeepSeek-V3"


def test_summarize_dedup_keeps_last_run():
    """旧冒烟行 + 重跑行同 probe_id，只算最后一次，保证每档干净 20 行。"""
    old = {
        "llm_model": "deepseek-ai/DeepSeek-V3", "run_tag": "deepseek/t9",
        "probe_id": 0, "prompt_tokens": 999, "completion_tokens": 99,
        "final_ok": True, "repair_count": 0, "latency_ms": 9999.0,
        "output": {"topic": "unknown"},
    }
    new = {
        **old,
        "run_tag": "deepseek-ai/DeepSeek-V3/t9",
        "prompt_tokens": 970, "completion_tokens": 41, "latency_ms": 3000.0,
        "output": {"topic": "hulu"},
    }
    s = summarize([old, new], gold_path=Path("/nonexistent/gold.json"))
    d = s["deepseek-ai/DeepSeek-V3"]
    assert d["n"] == 1
    assert d["total_cost"] == pytest.approx(0.002268, abs=1e-9)
    assert d["unknown_rate"] == 0.0
    assert d["accuracy"] is None  # gold 不存在时正确率留空，不假装 0


def test_four_models_registered_and_names_match_entities():
    """四档名实相符：Qwen7B / DeepSeek / Qwen72B / 混元 Hy4 各占一行，7B 未核价。"""
    assert set(MODEL_PRICING) == {
        "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-72B-Instruct",
        "tencent/Hy4-preview",
    }
    assert MODEL_PRICING["Qwen/Qwen2.5-7B-Instruct"] is None  # 不臆造单价


def test_hy4_price_matches_official_console():
    """Hy4 单价来自控制台截图 + 官网 pricing 页（¥6/¥18 每 M = 0.006/0.018 每 K）。"""
    assert MODEL_PRICING["tencent/Hy4-preview"] == (0.006000, 0.018000)
    # 用冒烟实测 token 对账：978 input / 841 output（含 740 reasoning）
    assert cost(978, 841, MODEL_PRICING["tencent/Hy4-preview"]) == pytest.approx(
        (978 * 0.006 + 841 * 0.018) / 1000, abs=1e-9
    )


def test_input_price_dominates_cheapness_order():
    """四档中已核价的 3 档：分类任务 input 占 ~95%，便宜序由 input 单价决定
    DeepSeek(0.002) < Qwen72B(0.00413) < Hy4(0.006)。7B 未核价不参与排序。"""
    p, c = 960, 43
    ds = cost(p, c, MODEL_PRICING["deepseek-ai/DeepSeek-V3"])
    qw = cost(p, c, MODEL_PRICING["Qwen/Qwen2.5-72B-Instruct"])
    hy = cost(p, c, MODEL_PRICING["tencent/Hy4-preview"])
    assert ds < qw < hy


def test_unpriced_model_is_not_silently_free():
    """单价未核对的档：row_cost=None、成本列全 None，绝不静默按 0 元算进成本表。"""
    cheap = "Qwen/Qwen2.5-7B-Instruct"
    row = {
        "llm_model": cheap, "run_tag": f"{cheap}/t9", "probe_id": 0,
        "prompt_tokens": 970, "completion_tokens": 41,
        "final_ok": True, "repair_count": 0, "latency_ms": 3000.0,
        "output": {"topic": "hulu"},
    }
    assert MODEL_PRICING[cheap] is None
    assert row_cost(row) is None
    d = summarize([row], gold_path=Path("/nonexistent/gold.json"))[cheap]
    assert d["priced_n"] == 0
    assert d["avg_cost"] is None and d["month_cost"] is None and d["year_cost"] is None
