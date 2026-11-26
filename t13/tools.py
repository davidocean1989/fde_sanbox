TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气状况和温度，当用户询问某地现在冷热/天气如何时使用。历史天气，未来多日预报/气候常识问题不要使用。字段city 为城市中文名，必填项。字段unit 为温标，非必填项。示例：用户问[北京热不热]时，city 填写'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，例如：北京，上海。不要填省份或国家",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["C", "F"],
                        "description": "温标，只能是 C 或F。用户未明确要求时不要填",
                    },
                },
                "required": ["city"],
            },
        },
    }
]
