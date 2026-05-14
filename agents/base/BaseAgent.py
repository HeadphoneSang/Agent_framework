from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Iterator

from config import BaseConfig
from internals import HelloAgentsLLM, Message
from config import AgentConfig
from logger.loggerUtil import get_logger


class BaseAgent(ABC):
    """
    基础的智能体基类
    """

    def __init__(self, name: str, llm: HelloAgentsLLM, system_prompt: Optional[str] = None,
                 config: Optional[BaseConfig] = None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or AgentConfig()
        self.logger = get_logger(name)
        self.temperature = self.config.get('temperature', 0.7)

    @abstractmethod
    def stream(self, input_params: Dict[str, Any], **kwargs) -> Iterator[Message]:
        """
        和智能体对话，同时返回一个迭代器，通过迭代器可以获得对话中的详细的每次对话记录
        :param input_params: 对话的kV字典，里面是提示词模板的格式化字典
        :return 消息迭代器
        """
        raise NotImplementedError

    def invoke(self, input_params: Dict[str, Any], **kwargs) -> list[Message]:
        """
        执行一次智能体会话，传入的kv对格式化自定义提示词模板，返回此次会话所有时间步的聊天记录
        :param input_params: kv通配符字典
        :param kwargs: 额外参数
        :return: 返回所有聊天的消息
        """
        session_msg: list[Message] = []
        msg_stream: Iterator[Message] = self.stream(input_params, **kwargs)
        for msg in msg_stream:
            session_msg.append(msg)
        return session_msg

    def __str__(self):
        return f"Agent(name={self.name}, model={self.llm.model})"
