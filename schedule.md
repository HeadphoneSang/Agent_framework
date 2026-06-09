# 任务列表：
- 自定义一个状态对象，包含status，和payload **(√)**
- 给HelloLLMS添加一个自定义调用stream或者是normal的接口，或者是改现在的，可以传入调用参数。**(√)**
- 完善FunctionalCallAgent，测试一下toolcall调用可以吗（）
- 测试一下toolcall的非流式，调用qwen可以吗（）
- 添加一个新的md文件，里面是一个简短的提示词，就是让大模型输出json格式的回复，方便tool_call进行。同时这个提示词只加入到第一次请求的messages里面，和真实的业务messages解耦，防止污染后续的自然语言聊天()
- 用Pydantic的BaseModel去做json结果封装, 同时考虑两层设计，如果llm支持，就走json_schema，否则就走普通json()
```python
if self.model_support_json_schema:

    response = client.beta.chat.completions.parse(
        ...
    )

else:

    response = client.chat.completions.create(
        ...
    )

    content = response.choices[0].message.content

    result = repair_and_parse_json(content)
```
- 判断模型是否支持的方案: 
1. 白名单文件
2. try_exception
```python
try:
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=PlannerResult
    )

except Exception as e:

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    content = response.choices[0].message.content
    result = repair_and_parse_json(content)
```
# ToolCallAgent设计方案：
## 1. 通过强制模型调用一个工具，然后工具的参数里面要求填入你想要的json的格式，这样可以100%强制返回固定格式的json
### 示例代码：
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
                "analysis": {"type": "string", "description": "思考过程"},
                "need_tool": {"type": "boolean"},
                "plan": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["analysis", "need_tool", "plan"]
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
## 2. 通过开启response_format然后通过提示词强制要求json格式来获取json结果
### 示例代码：
```python
import json
from openai import OpenAI

client = OpenAI(api_key="your_deepseek_key", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system", 
            # 必须在 prompt 中明确提及 JSON 格式，并最好给出结构样例
            "content": "你是一个工具调用规划助手。请分析用户问题并决定是否使用工具。请务必返回 JSON 格式，结构如下：{'need_thought': bool, 'plan': list}"
        },
        {"role": "user", "content": "帮我查一下今天北京的天气。"}
    ],
    # 🌟 开启 JSON 模式
    response_format={
        'type': 'json_object'
    },
    temperature=0.3 # 建议调低温度，输出更稳定
)

# 获取字符串
json_str = response.choices[0].message.content
print("原始字符串:", json_str)

# 解析成 Python 字典
data = json.loads(json_str)
print("解析后的字典:", data)
```
- 优点： 支持度高，但是需要系统提示词里面给具体的json结构示例，而且**一定要加上json字段在消息**