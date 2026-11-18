from openai import OpenAI
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL_DEEPSEEK

_client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)


def real_llm_complete(messages: list[dict], temperature: float):
    resp = _client.chat.completions.create(
        model=SILICONFLOW_MODEL_DEEPSEEK,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,
    )
    print(resp.model)
    print(
        "\n使用量:\n",
        "prompt_tokens:",
        resp.usage.prompt_tokens,
        "\n",
        "completion_tokens:",
        resp.usage.completion_tokens,
        "\n",
        "total_tokens:",
        resp.usage.total_tokens,
    )

    return resp.choices[0].message.conatent or ""  # 空回复兜底，别把 None 交给 parse
