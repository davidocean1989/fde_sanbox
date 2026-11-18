# parse.py  -- 纯函数层：不碰网路，不碰 message，不写文件
from pydantic import ValidationError
from schema import TopicResult
import json

# 错误类型
_KIND = {
    "json_invalid": "syntax",  # fence / 单引号/尾逗号/解释文字
    "literal_error": "enum",  # 自造主题池
    "missing": "missing",  # 缺字段
    "string_too_short": "empty",  # reason空壳
    "less_than_equal": "range",
    "greater_than_equal": "range",
}


class ParseError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


# 转换，要么成功后的 json ，要么返回错误代码
def parse(raw: str) -> TopicResult:
    """L1 + L2 合一：不可信字符串 ->可信对象。全项目唯一的信任转换入口"""
    cleaned = _strip_fence(raw)
    try:
        return TopicResult.model_validate_json(cleaned)
    except ValidationError as e:
        errs = e.errors()
        code = errs[0]["type"] if errs else "unknown"
        raise ParseError(_KIND.get(code, "type"), str(e)) from e


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
                return text[start : i + 1]
    return None


# 去除 fence
def _strip_fence(text: str) -> str:
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        candidate = _scan_json(text, start)
        if candidate is None:
            continue
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    return text.strip()
