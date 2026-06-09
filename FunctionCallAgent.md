# ToolCallAgent设计方案：
## 首先通过一个固定格式的function_call发起一个LLM调用，要求LLM必须调用这个函数。然后，这个函数的参数包含了LLM对于当前问题的思考文本、是否需要调用工具。然后：
- 如果需要调用工具，就根据前文的分析，直接发起一个普通的toolcall的llm调用。然后执行工具返回结果，并且将结果封装 `BaseState`。其中，payload是结果，状态是`StateCode.TOOL_CALL`
- 如果不需要调用工具，则直接返回结果，并将结果封装`BaseState`，其中，payload是结果，状态是`StateCode.SUCCESS`
### 示例函数格式：
```python
# 1. 定义一个你期望的格式工具（Schema）
my_tools = [{
    "type": "function",
    "function": {
        "name": "final_structured_output",
        "description": "用于输出最终的规划结果",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "思考过程，比如对于当前问题的回答或者是解决问题的思路。该分析可以直接根据已有信息，回答问题，也可以为下文的具体工具调用提供调用的信息。"},
                "need_tool": {"type": "boolean"}}
            },
            "required": ["analysis", "need_tool"]
        }
    }
}]

# 2. 发起请求
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "帮我查一下今天北京的天气。"}],
    tools=my_tools,
    # 🌟 强行指定模型必须调用这个函数
    tool_choice={"type": "function", "function": {"name": "final_structured_output"}}
)

# 3. 提取参数（绝对是标准的 JSON 格式）
tool_call = response.choices[0].message.tool_calls[0]
json_arguments = tool_call.function.arguments

import json
print(json.loads(json_arguments))
```
- 优点： > 100% 稳定，哪怕模型想废话聊天，网关也会逼着它只能往 arguments 里填 JSON。