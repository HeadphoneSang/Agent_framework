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
    name="写作机器人",
    llm=llm,
    config=AgentConfig(),
    tool_registry=tool_register
)
final_answer: Message = agent.run(stream=True, input_msg=Message(role="user",content="帮我规划一下，今年暑假去张家界的计划"))
get_logger().info(final_answer.content)