from abc import ABC, abstractmethod
from asyncio import Condition
from typing import Any, Dict, List

from agents.base import BaseAgent
from internals import Message, BaseMemory


class BaseMemoryAgent(BaseAgent, ABC):
    """
    有持久化内存的智能体
    """

    # 持久化记忆，需要读写保护
    session_memories: Dict[str, BaseMemory] = {}

    @abstractmethod
    def _init_memory(self, config: Dict[str, Any]) -> BaseMemory:
        """
        为新的用户会话创建一个持久化记忆
        :param config: 用户会话的参数
        :return: 返回一个BaseMemory
        """
        pass

    def run_w_memory(self, input_msg: Message, config: Dict[str, Any] = None, **kwargs) -> Message:
        """
        带有持久化历史记录的会话
        :param input_msg: 用户的提问
        :param config: 用户的会话参数
        :return: 返回智能体的回答
        """
        session_id = config.get("session_id", "default")
        memory = self.session_memories.get(session_id, None)
        if memory is None:
            memory = self._init_memory(config)
            self.session_memories[session_id] = memory
        persist_memories: List[Message] = memory.get_memories()
        # 将持久化内存添加到会话历史中
        for message in persist_memories:
            self.add_history(message)
        assistant_msg: Message = self.run(input_msg, **kwargs)
        memory.add_memories([input_msg, assistant_msg])
        return assistant_msg

