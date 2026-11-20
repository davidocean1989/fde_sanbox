import time
from typing import Callable
from openai import OpenAI
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL_DEEPSEEK
from contracts import LLMResult, Usage

_client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)


def real_llm_complete(
    messages: list[dict], temperature: float, model: str | None = None
) -> LLMResult:
    """真实 API 调用。返回具名结果对象，不在这一层把 usage 压成一个 int。

    model 可选：不传就用 config 里的默认档，传了就跑指定档（T9 四档对比用）。
    """
    model_name = model or SILICONFLOW_MODEL_DEEPSEEK
    started = time.perf_counter()
    resp = _client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=5000,
    )

    # 逐字段兜底：choices 可能为空，content 可能是 None，usage 可能整个缺失
    choice = resp.choices[0] if resp.choices else None
    content = (choice.message.content or "") if choice else ""
    u = resp.usage
    usage = (
        Usage(
            prompt_tokens=getattr(u, "prompt_tokens", None),
            completion_tokens=getattr(u, "completion_tokens", None),
            total_tokens=getattr(u, "total_tokens", None),
            raw=u.model_dump() if hasattr(u, "model_dump") else None,
        )
        if u is not None
        else None
    )

    result = LLMResult(
        content=content,
        usage=usage,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        # resp.model 是服务端实际响应的模型（可能带版本号），比请求名更可信
        model=resp.model or model_name,
        finish_reason=choice.finish_reason if choice else None,
    )
    # 仅调试观察。统计口径以 result.usage 为准，不参与落盘
    print(f"[client] {result.model} usage={usage} finish={result.finish_reason}")
    return result


def make_llm_complete(model_name: str) -> Callable[[list[dict], float], LLMResult]:
    """把模型名绑进闭包，LlmFn 签名仍是 (messages, temperature)，不必改接缝。

    T9 用法：main(make_llm_complete("Qwen/Qwen2.5-72B-Instruct"), ...)
    """

    def _fn(messages: list[dict], temperature: float) -> LLMResult:
        return real_llm_complete(messages, temperature, model=model_name)

    return _fn
