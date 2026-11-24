import time

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL_DEEPSEEK,
)
from contracts import LLMResult, Usage
from openai import OpenAI

_client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)


def real_llm_complete(
    messages: list[dict], temperature: float, model: str | None = None
):
    model_name = model or SILICONFLOW_MODEL_DEEPSEEK
    started = time.perf_counter()
    resp = _client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=5000,
        extra_body={"enable_thinking": False},
    )
    # 逐字段兜底：choices 可能为空，content 可能是 None ，usage 可能整个缺失
    choice = resp.choices[0] if resp.choices else None
    content = (choice.message.content or "") if choice else ""
    u = resp.usage

    usage = (
        Usage(
            # 对不确定对象的处理方式，使用getattr和hasattr。属于try/except AttributeError的缩写
            prompt_tokens=getattr(u, "prompt_tokens", None),
            completion_tokens=getattr(u, "completion_tokens", None),
            total_tokens=getattr(u, "total_tokens", None),
            raw=u.model_dump() if hasattr(u, "c") else None,
        )
        if u is not None
        else None
    )

    result = LLMResult(
        content=content,
        usage=usage,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        # resp.model 是服务端实际响应的模型，比请求更可信
        model=resp.model or model_name,
        finish_reason=choice.finish_reason if choice else None,
    )
    print(f"client={result.model} usage = {usage} finish={result.finish_reason}")
    return result


def make_llm_complete(model_name: str):
    """把模型名绑进闭包，LlmFn 签名仍是（messages,temperature)，不必改接缝。
    T9 用法：main(make_llm_complete("Qwen/..."),...)
    """

    def _fn(messages: list[dict], temperature: float):
        return real_llm_complete(messages, temperature, model=model_name)

    return _fn
