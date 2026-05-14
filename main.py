from typing import Dict, Any

from agents import ReflectAgent, SimplePlannerAgent, ReactAgent
from config import AgentConfig
from externals.memory.RedisMemory import RedisMemory
from internals import Message
from internals.memory import BaseMemory, MemoryInMemory
from internals.tool import ToolRegistry
from llms.MyLLM import MyLLM
from logger.loggerUtil import get_logger
from tools.local_tools import CalculateTool, TimeTool
from tools.web_tools import SearchTool
from internals.memory import AgentWithMemoryProxy


def init_memory(config: Dict[str, Any]) -> BaseMemory:
    return RedisMemory(config)


llm = MyLLM(provider='qwen', print_content=True)

tool_register = ToolRegistry()

tool_register.register(CalculateTool())
tool_register.register(TimeTool())
tool_register.register(SearchTool())

# agent = SimplePlannerAgent("规划机器人", llm, tool_registry=tool_register,config=AgentConfig())
#
# msg: Message = agent.run(input_msg=Message(role="user", content="请帮计算一下6^2+(19*2)的结果"),stream=True)
# print(msg.content)

agent = ReactAgent(
    name="问答机器人",
    llm=llm,
    config=AgentConfig(),
    tool_registry=tool_register
)
agent = AgentWithMemoryProxy(agent=agent, init_memory_func=init_memory)
stream_res = agent.stream(input_params={"question": "还有什么别的建议吗？"},
                          stream=True)
for msg in stream_res:
    print("============================")
    print(str(msg))

# memory_proxy = AgentWithMemoryProxy(agent, init_memory_func=init_memory)
#
# answer_0: Message = memory_proxy.run(
#     input_msg=Message(role="user", content="你还有什么建议吗？"),
#     session_config={"session_id": "12345"}, stream=True, )

# agent = ReactAgent("普通机器人", llm=llm,tool_registry=tool_register)
# history = []
# response_stream = agent.run_with_format(input_params={"history": history, "question": "帮我规划一下明天去北京的衣服"},track_msg=True,stream=True)
#
# for response_msg in response_stream:
#     print("============================")
#     print(str(response_msg))
