from agents import ReactAgent, ReflectAgent
from config import AgentConfig
from internals import Message
from internals.tool import ToolRegistry
from llms.MyLLM import MyLLM
from logger.loggerUtil import get_logger
from tools.local_tools import CalculateTool, TimeTool
from tools.web_tools import SearchTool

llm = MyLLM(provider='qwen')

tool_register = ToolRegistry()

tool_register.register(CalculateTool())
tool_register.register(TimeTool())
tool_register.register(SearchTool())

agent = ReflectAgent(
    name="问答机器人",
    llm=llm,
    config=AgentConfig(),
    tool_registry=tool_register
)
answer_0: Message = agent.run_w_memory(input_msg=Message(role="user",content="这两天北京的天气怎么样？不用太严谨，直接回答给我就行"),config={"session_id":"12345"},stream=True,)
get_logger().info(answer_0.content)
answer_1: Message = agent.run_w_memory(input_msg=Message(role="user",content="那我需要拿什么衣物？"),config={"session_id":"12345"},stream=True,)
