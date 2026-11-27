"""四个设计决策
1. MAX_STEPS 硬上限：没有上限的 Agent 遇到「工具一直返回错误」会无限循环。真实事故里烧掉几百美元的案例很多。
2. 工具异常不抛出，转成字符串喂回去：Agent 需要「看到」错误才能自我修正，抛异常等于直接终止。
3. 结果截断到 4000 字符：某个工具返回一个 10 万字的表，一次就把上下文撑爆。
4. 走满步数要标记 truncated：这跟「正常结束」必须区分，否则你会把失败统计成成功。"""

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL_DEEPSEEK,
    SILICONFLOW_MODEL_QWEN72B,
)
from openai import OpenAI
from tools import TOOLS

probe = "今天北京的天气怎么样？热不热。"


def fake_weather():
    # return '{"error":"参数错误，请重新调用 get_weather","retry":true}'
    return '{"city":"北京","temp":38,"humidity":60,"condition":"晴"}'


def runmodel(model: str, probe: str):
    client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
    message = [{"role": "user", "content": probe}]

    tool_calls_id = None
    step = 0
    MAX_STEPS = 5
    status = "truncated"
    while step < MAX_STEPS:
        step += 1
        resp = client.chat.completions.create(
            model=model, tools=TOOLS, messages=message, tool_choice="auto"
        )
        choice = resp.choices[0]
        finish_reason = choice.finish_reason
        print("第", step, "轮:")
        print("finish_reason: ", finish_reason)
        print(
            "messages的长度：", len(message), "message[-1]的类型：", type(message[-1])
        )
        print("messages:", message)
        if finish_reason == "stop":
            status = "stop"
            print("stop，对应的choice.message.content是", choice.message.content)
            break
        func_name = ""

        if finish_reason == "tool_calls":
            tool_calls_id = choice.message.tool_calls[-1].id
            func_name = choice.message.tool_calls[-1].function.name

        content = fake_weather() if func_name == "get_weather" else ""

        message.append(choice.message)

        message.append(
            {"role": "tool", "tool_call_id": tool_calls_id, "content": content}
        )
    print("运行结束，status 状态是：", status)


def main():
    runmodel(SILICONFLOW_MODEL_DEEPSEEK, probe)


if __name__ == "__main__":
    main()
