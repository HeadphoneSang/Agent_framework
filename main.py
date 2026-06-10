from typing import Dict, Any

from agents import ReflectAgent, SimplePlannerAgent, ReactAgent, ReactTCAgent
from config import AgentConfig
from externals.memory import RedisMemory,LocalFileMemory
from internals import Message
from internals.memory import BaseMemory, MemoryInMemory
from internals.tool import ToolRegistry
from llms.MyLLM import MyLLM
from logger.loggerUtil import get_logger
from tools.local_tools import CalculateTool, TimeTool
from tools.web_tools import SearchTool
from internals.memory import AgentWithMemoryProxy


def init_memory(config: Dict[str, Any]) -> BaseMemory:
    return LocalFileMemory(config)


llm = MyLLM(provider='qwen', print_content=True)

tool_register = ToolRegistry()

tool_register.register(CalculateTool())
tool_register.register(TimeTool())
tool_register.register(SearchTool())

agent = ReactTCAgent(
    name="问答机器人",
    llm=llm,
    config=AgentConfig(),
    tool_registry=tool_register
)
agent = AgentWithMemoryProxy(agent=agent, init_memory_func=init_memory)
stream_res = agent.stream(input_params={"question": "美国的民主共和和朝鲜的社会主义有什么区别，查一下资料，给我详细的答案"},
                          stream=True)
for msg in stream_res:
    print("============================")
    print(str(msg))
