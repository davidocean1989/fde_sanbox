from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL_DEEPSEEK,
    SILICONFLOW_MODEL_QWEN72B,
)
from openai import OpenAI
from tools import TOOLS

client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
probe = "今天北京的天气怎么样？热不热。"
message = [{"role": "user", "content": probe}]

resp = client.chat.completions.create(
    model=SILICONFLOW_MODEL_QWEN72B, tools=TOOLS, messages=message, tool_choice="auto"
)

print(resp.model_dump_json(indent=2))
