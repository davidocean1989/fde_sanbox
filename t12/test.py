import json


def _fence_(text: str):
    for start, ch in enumerate(text):
        if ch not in "[{":
            continue
        candination = _jsonget(start, text)
        if candination is None:
            continue
        try:
            json.loads(candination)
        except json.JSONDecodeError:
            continue
        return candination
    return text.strip()


def _jsonget(start: int, text: str):
    in_string = False
    escape = False
    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                return None
            elif (stack.pop(), ch) not in [("{", "}"), ("[", "]")]:
                return None
            elif not stack:
                return text[start : i + 1]


a = '{"a": "x}y", "b": [1,2],"c":123}'
b = '{"a": [1,2], "b": 3}'
c = '{"a": "x}y"}'
d = '{"k": "a\\"b"}'
e = '```json\n{"a":1}\n```'
f = '好的，结果如下：\n```json\n{"a":1}\n```\n希望有帮助'
print(_fence_(f))
