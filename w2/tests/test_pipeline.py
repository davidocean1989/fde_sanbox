import pytest
from pipeline import extract_one, RepaireExhausted
from parse import ParseError

GOOD = '{"topic":"hulu","confidence":0.8,"reason":"记录的是 Hululu Garden 的主页内容"}'


def fake_llm(script):  # 工厂：给它一份剧本，它造出一个假模型 + 一个记录本
    calls = []  # 闭包共享的记录本（spy 通道）

    def _fn(messages, temperature):
        calls.append([dict(m) for m in messages])  # 快照存档
        return script[min(len(calls) - 1, len(script) - 1)]  # 查剧本，返回

    return _fn, calls  # 同时交出「控制权」和「观察权」


def test_act1_clean_output_parse():
    script_act1 = [
        "```json\n" + GOOD + "\n```",
    ]

    fn, calls = fake_llm(script_act1)
    probe = "this is a probe"
    r = extract_one(probe, fn, max_repair=1)
    assert type(r) == dict


def test_act2_bad_then_good_repairs():
    script_act2 = [
        "```json\n{'topic': 'hulu',}\n```",
        GOOD,
    ]
    probe = "this is a probe"
    fn, calls = fake_llm(script_act2)
    r = extract_one(probe, fn, max_repair=1)


def test_act3_both_bad_raises():
    script_act3 = [
        "{'topic': 'hulu',}",
    ]
    probe = "this is a probe"
    fn, calls = fake_llm(script_act3)
    with pytest.raises(RepaireExhausted):
        extract_one(probe, fn, max_repair=1)
