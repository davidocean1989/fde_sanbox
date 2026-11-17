import w2.main1 as main1


test_a = """```json\n{"theme":"rag"}\n```"""
test_b = """```\n{"a":1}\n```"""
test_c = """以下是结果：{"a":1} 希望帮到你"""
test_d ="""{"a":1}"""
test_e = """```[{"a":1}]"""
test_f = """```json\n[{"a":1},{"b":2}]\n```"""
test_g = ""
test_h = """结果(第{1}条)：{"a":1}"""
test_i = """{"a":1} 备注：{说明}"""

def test_strip_fence():
    assert main1._strip_fence(test_a) == '{"theme":"rag"}'
    assert main1._strip_fence(test_b) == '{"a":1}'
    assert main1._strip_fence(test_c) == '{"a":1}'
    assert main1._strip_fence(test_d) == '{"a":1}'
    assert main1._strip_fence(test_e) == '[{"a":1}]'
    assert main1._strip_fence(test_f) == '[{"a":1},{"b":2}]'
    assert main1._strip_fence(test_g) == ''
    assert main1._strip_fence(test_h) == '{"a":1}'
    assert main1._strip_fence(test_i) == '{"a":1}'