from abc import ABC, abstractmethod
from typing import Optional, List

from config import BaseConfig
from internals import HelloAgentsLLM, Message
from config import AgentConfig


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
        # 智能体聊天历史
        self.history: List[Message] = []

    @abstractmethod
    def run(self, input: Message, **kwargs) -> Message:
        """启动Agent"""
        pass

    def add_history(self, message: Message):
        """添加历史"""
        self.history.append(message)

    def clear_history(self):
        """清空历史"""
        self.history.clear()

    def get_history(self) -> List[Message]:
        """获取历史"""
        return self.history.copy()

    def __str__(self):
        return f"Agent(name={self.name}, model={self.llm.model})"
