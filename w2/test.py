test_a = """```json\n{"theme":"rag"}\n```"""
test_b = """```\n{"a":1}\n```（无语言标记）"""
test_c = """以下是结果：{"a":1} 希望帮到你（无围栏）"""
test_d ="""{"a":1}（本来就干净）"""


def _strip_fence(text: str) -> str:
    if text.count("```")==2:
        text = text.split("```")[1]  # 

# 考虑到s[-1:0] 不报错，返回空串的情况
    if text.find('{') != -1 and text.find('}') != -1:
        text = text[text.find('{'):text.rfind('}')+1]  # 提取 JSON 内容
    return text.strip()
    text = text[text.find('{'):text.rfind('}')+1]  # 提取 JSON 内容
        
    return text.strip()    

print(_strip_fence(test_a))
print(_strip_fence(test_b))
print(_strip_fence(test_c)) 
print(_strip_fence(test_d))