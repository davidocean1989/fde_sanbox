import pytest
from dataclasses import asdict
from pipeline import extract_one, RepaireExhausted
from parse import ParseError
from prompts import SYSTEM_PROMPT_HASH
from contracts import LLMResult, Usage

GOOD = '{"topic":"hulu","confidence":0.8,"reason":"记录的是 Hululu Garden 的主页内容"}'


def fake_llm(
    script, usage_script=None, model="fake"
):  # 工厂：给它一份剧本，它造出一个假模型 + 一个记录本
    calls = []  # 闭包共享的记录本（spy 通道）

    def _fn(messages, temperature):
        calls.append([dict(m) for m in messages])  # 快照存档
        idx = min(len(calls) - 1, len(script) - 1)  # 按调用序号取，剧本用完才钳住
        if usage_script is None:
            usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        elif not usage_script:  # 空剧本 guard：否则 min(idx, -1) 仍会 IndexError
            usage = None
        else:
            usage = usage_script[min(idx, len(usage_script) - 1)]
        return LLMResult(  # 与真 client 同一个返回契约，不再是 [text, 0]
            content=script[idx],
            usage=usage,
            latency_ms=1.0,
            model=model,
            finish_reason="stop",
        )

    return _fn, calls  # 同时交出「控制权」和「观察权」


def test_act1_clean_output_parse():
    script_act1 = [
        "```json\n" + GOOD + "\n```",
    ]

    fn, calls = fake_llm(script_act1)
    probe = "this is a probe"
    r = extract_one(probe, fn, max_repair=1)
    # 出口 1：一次过，不该有第二次调用
    assert r.final_ok is True
    assert r.repair_count == 0
    assert r.output["topic"] == "hulu"
    assert r.error is None
    assert r.error_kind is None
    assert r.error_history == []
    assert len(calls) == 1
    assert len(calls[0]) == 2
    # 一轮调用：token 就是这一轮的值
    assert r.prompt_tokens == 10
    assert r.completion_tokens == 5
    assert r.total_tokens == 15
    assert r.usage_missing == 0
    # 实验元数据：模型与 prompt 版本必须落盘
    assert r.llm_model == "fake"
    assert r.prompt_hash == SYSTEM_PROMPT_HASH


def test_act2_bad_then_good_repairs():
    script_act2 = [
        "```json\n{'topic': 'hulu',}\n```",
        GOOD,
    ]
    probe = "this is a probe"
    fn, calls = fake_llm(script_act2)
    r = extract_one(probe, fn, max_repair=1)
    # 出口 2：修一次过；第二次调用必须带着坏输出回喂
    assert r.final_ok is True
    assert r.repair_count == 1
    assert r.output["topic"] == "hulu"
    assert r.error is None

    assert len(r.error_history) == 1
    assert len(calls) == 2
    assert len(calls[1]) == 4
    assert calls[1][2]["content"] == script_act2[0]
    # 两轮调用：总体 token 必须是累加值，不是「最后一次」
    assert r.total_tokens == 30
    assert r.prompt_tokens == 20
    assert r.completion_tokens == 10
    assert r.usage_missing == 0


def test_act3_both_bad_raises():
    # script_act3 = [
    #     "{'topic': 'hulu',}",
    # ]
    script_act3 = ["""{"theme":"hulu","confidence":0.9,"reason":"..."} """]
    probe = "this is a probe"
    fn, calls = fake_llm(script_act3)

    with pytest.raises(RepaireExhausted) as exc:
        extract_one(probe, fn, max_repair=1)
    r = exc.value.result
    assert r.error_kind == "missing"
    assert r.repair_count == 1
    assert r.output is None
    assert len(calls) == 2
    assert r.error is not None
    assert r.final_ok is False
    assert len(r.error_history) == 2
    # 失败的两轮同样要记账，不能因为没成功就丢掉 token
    assert r.total_tokens == 30
    # 出口 3 也必须带上实验元数据，否则失败行无法归属
    assert r.llm_model == "fake"
    assert r.prompt_hash == SYSTEM_PROMPT_HASH


def test_usage_missing_counted_separately():
    """拿不到 usage 记 None + 单独计数，不能假装成 0 token 混进成本表。"""
    fn, calls = fake_llm(
        ["```json\n{'topic': 'hulu',}\n```", GOOD],
        usage_script=[
            None,
            Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        ],
    )
    r = extract_one("this is a probe", fn, max_repair=1)
    assert r.final_ok is True
    assert r.repair_count == 1
    assert r.total_tokens == 10  # 只累加拿得到的那一轮
    assert r.usage_missing == 1  # 缺失轮次被标记，stats 层可剔除
    # usage 缺失不影响 model 落盘
    assert r.llm_model == "fake"


def test_four_models_are_distinguishable_in_rows():
    """T9 四档 80 行落进同一个 jsonl，靠 llm_model 分组；prompt_hash 必须全同。"""
    names = [
        "Qwen/Qwen2.5-7B-Instruct",
        "zai-org/GLM-4.5-Air",          # 替 Hy4，须关 thinking
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-72B-Instruct",
    ]
    rows = []
    for name in names:
        fn, _ = fake_llm([GOOD], model=name)
        rows.append(asdict(extract_one("probe", fn, max_repair=0)))
    assert {r["llm_model"] for r in rows} == set(names)
    assert len({r["prompt_hash"] for r in rows}) == 1  # 控制变量：同一版强 prompt


def test_temperature_is_passed_through():
    """温度验证 0 / 0.3 / 0.8 的前提：temperature 不能再写死在 extract_one 里。"""
    seen = []

    def _fn(messages, temperature):
        seen.append(temperature)
        return LLMResult(
            content=GOOD,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="fake",
        )

    extract_one("probe", _fn, max_repair=0, temperature=0.8)
    assert seen == [0.8]


def test_temperature_is_persisted_in_result():
    """温度是自变量，必须显式落盘；否则 0 / 0.3 / 0.8 三批从数据里分不开。"""
    fn, _ = fake_llm([GOOD])
    r = extract_one("probe", fn, max_repair=0, temperature=0.3)
    assert r.temperature == 0.3
    assert asdict(r)["temperature"] == 0.3
    # 三个出口（一次过 / 修一次过 / 耗尽）都得带，主实验不传时是 0.0
    fn0, _ = fake_llm([GOOD])
    assert extract_one("probe", fn0, max_repair=0).temperature == 0.0
