from openai import OpenAI
from prompts import CLASSIFIER_SYSTEM_PROMPT,BARE_PROMPT_HASH
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL_DEEPSEEK, PROMPT_CHINESE,PROMPT_ENGLISH, PROMPT_EASY, SILICONFLOW_MODEL_QWEN,  PROMPT_CHINESE_B,PROMPT_TRAP, PROMPT_TEST1, PROMPT_TEST2,PROMPT_TEST3,PROMPT_TEST4
client = OpenAI(api_key=SILICONFLOW_API_KEY,base_url=SILICONFLOW_BASE_URL)
response = client.chat.completions.create(
    temperature=0,
    model=SILICONFLOW_MODEL_DEEPSEEK,
    messages=[{"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT}, {"role": "user", "content": PROMPT_TEST4} ],
)
print(response.model)
print("\n返回内容：\n",response.choices[0].message.content)
print("\n使用量:\n", "prompt_tokens:", response.usage.prompt_tokens,"\n", "completion_tokens:", response.usage.completion_tokens, "\n", "total_tokens:", response.usage.total_tokens)

