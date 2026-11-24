from pathlib import Path

import pytest
from pricing import (
    MODEL_PRICING,
    PRICE_DIVISOR,
    PRICE_UNIT,
    T9_MODELS,
    cost,
    resolve_model,
    row_cost,
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
    new = {
        "llm_model": "resp-name-with-suffix",
        "run_tag": "deepseek-ai/DeepSeek-V3/t9",
    }
    old = {"llm_model": "deepseek-ai/DeepSeek-V3", "run_tag": "deepseek/t9"}
    assert resolve_model(new) == "deepseek-ai/DeepSeek-V3"
    assert resolve_model(old) == "deepseek-ai/DeepSeek-V3"


def test_summarize_dedup_keeps_last_run():
    """旧冒烟行 + 重跑行同 probe_id，只算最后一次，保证每档干净 20 行。"""
    old = {
        "llm_model": "deepseek-ai/DeepSeek-V3",
        "run_tag": "deepseek/t9",
        "probe_id": 0,
        "prompt_tokens": 999,
        "completion_tokens": 99,
        "final_ok": True,
        "repair_count": 0,
        "latency_ms": 9999.0,
        "output": {"topic": "unknown"},
    }
    new = {
        **old,
        "run_tag": "deepseek-ai/DeepSeek-V3/t9",
        "prompt_tokens": 970,
        "completion_tokens": 41,
        "latency_ms": 3000.0,
        "output": {"topic": "hulu"},
    }
    s = summarize([old, new], gold_path=Path("/nonexistent/gold.json"))
    d = s["deepseek-ai/DeepSeek-V3"]
    assert d["n"] == 1
    assert d["total_cost"] == pytest.approx(0.002268, abs=1e-9)
    assert d["unknown_rate"] == 0.0
    assert d["accuracy"] is None  # gold 不存在时正确率留空，不假装 0


def test_t9_models_match_entities_and_are_all_priced():
    """本轮四档名实相符，且四档单价全部核对过（不再有 None / n/a）。"""
    assert T9_MODELS == (
        "Qwen/Qwen2.5-7B-Instruct",
        "zai-org/GLM-4.5-Air",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-72B-Instruct",
    )
    assert all(m in MODEL_PRICING and MODEL_PRICING[m] is not None for m in T9_MODELS)
    # 7B 是官方 Free：核对过的真实 0 元，不是「价格未知」的 None
    assert MODEL_PRICING["Qwen/Qwen2.5-7B-Instruct"] == (0.000000, 0.000000)


def test_hy4_excluded_from_t9_but_kept_for_audit():
    """Hy4 已剔除出本轮四档，但价格留在字典里，旧留档文件仍可重算。"""
    assert "tencent/Hy4-preview" not in T9_MODELS
    assert MODEL_PRICING["tencent/Hy4-preview"] == (0.006000, 0.018000)


def test_hy4_price_matches_official_console():
    """[审计] Hy4 单价来自控制台截图 + 官网 pricing 页（¥6/¥18 每 M = 0.006/0.018 每 K）。"""
    assert MODEL_PRICING["tencent/Hy4-preview"] == (0.006000, 0.018000)
    # 用冒烟实测 token 对账：978 input / 841 output（含 740 reasoning）
    assert cost(978, 841, MODEL_PRICING["tencent/Hy4-preview"]) == pytest.approx(
        (978 * 0.006 + 841 * 0.018) / 1000, abs=1e-9
    )


def test_glm_price_matches_console():
    """GLM-4.5-Air 单价来自 2026-11-20 控制台「在线推理」：0.001 / 0.006 元 / K。"""
    assert MODEL_PRICING["zai-org/GLM-4.5-Air"] == (0.001000, 0.006000)
    # 关思考后冒烟实测 1005 input / 42 output
    assert cost(1005, 42, MODEL_PRICING["zai-org/GLM-4.5-Air"]) == pytest.approx(
        (1005 * 0.001 + 42 * 0.006) / 1000, abs=1e-9
    )


def test_price_ladder_is_monotonic():
    """换掉 Hy4 后四档第一次构成单调价格阶梯（input 单价 0 → 0.001 → 0.002 → 0.00413）。
    分类任务 input 占 ~95%，所以用同一组 token 量验证总成本序即可。"""
    p, c = 960, 43
    vals = [cost(p, c, MODEL_PRICING[m]) for m in T9_MODELS]
    assert vals == sorted(vals)
    assert vals[0] == 0.0  # 7B 免费档
    assert len({round(v, 9) for v in vals}) == 4  # 四档互不重合


def test_free_model_is_priced_zero_not_na():
    """0 元 ≠ 缺失：核对过的免费档 row_cost 返回 0.0，成本列显示 0.000000 而非 n/a。
    真正的「未知模型 / 缺 token 字段」仍必须返回 None（见上面两条用例）。"""
    cheap = "Qwen/Qwen2.5-7B-Instruct"
    row = {
        "llm_model": cheap,
        "run_tag": f"{cheap}/t9",
        "probe_id": 0,
        "prompt_tokens": 970,
        "completion_tokens": 41,
        "final_ok": True,
        "repair_count": 0,
        "latency_ms": 3000.0,
        "output": {"topic": "hulu"},
    }
    assert row_cost(row) == 0.0
    d = summarize([row], gold_path=Path("/nonexistent/gold.json"))[cheap]
    assert d["priced_n"] == 1
    assert d["avg_cost"] == 0.0 and d["month_cost"] == 0.0 and d["year_cost"] == 0.0
