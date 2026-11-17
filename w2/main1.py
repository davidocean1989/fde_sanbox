import json


def _scan_json(text: str, start: int) -> str | None:
    """从 start 处的 { 或 [ 出发配平括号，返回候选串；配不平返回 None。"""
    stack = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            if (stack.pop(), ch) not in (("{", "}"), ("[", "]")):
                return None
            if not stack:
                return text[start:i + 1]
    return None



def _strip_fence(text: str) -> str:
    for start,ch in enumerate(text):
        if ch not in '{[':
            continue
        candidate = _scan_json(text,start)
        if candidate is None:
            continue
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    return text.strip()