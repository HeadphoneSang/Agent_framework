from abc import ABC, abstractmethod
from asyncio import Condition
from typing import Any, Dict, List

from agents.base import BaseAgent
from internals import Message
from internals.memory import BaseMemory

# 持久化记忆
session_memories: Dict[str, BaseMemory] = {}


class BaseMemoryAgent(BaseAgent, ABC):
    """
    有持久化内存的智能体
    """

    @abstractmethod
    def _init_memory(self, config: Dict[str, Any]) -> BaseMemory:
        """
        为新的用户会话创建一个持久化记忆
        :param config: 用户会话的参数
        :return: 返回一个BaseMemory
        """
        raise NotImplementedError

    def invoke(self, input_params: Dict[str, Any], question_key: str = "question", memories_key: str = "history",
               session_config=None, **kwargs) -> \
            list[Message]:
        if session_config is None:
            session_config = {"session_id": "default"}
        session_id = session_config.get("session_id", "default")
        memory = session_memories.get(session_id, None)
        if memory is None:
            memory = self._init_memory(session_config)
            session_memories[session_id] = memory
        persist_memories: List[Message] = memory.get_memories()
        # 将持久化内存添加到会话历史中
        history_prompt = "- " + "\n\n- ".join([f"[{msg.role}]: {msg.content}" for msg in persist_memories])
        input_params[memories_key] = history_prompt
        msg_list: list[Message] = super().invoke(input_params=input_params, **kwargs)
        question: str = input_params.get(question_key, "")
        memory.add_memories([Message(content=question, role="user"), msg_list[-1]])
        return msg_list
