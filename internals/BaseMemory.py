from abc import ABC, abstractmethod
from typing import List, Any, Dict

from internals import Message


class BaseMemory(ABC):

    def __init__(self, config: Dict[str, Any], **kwargs):
        self.config = config

    @abstractmethod
    def add_memory(self, message: Message) -> None:
        """
        添加一条持久化记忆
        :param message: 消息
        :return: 无
        """
        pass

    def add_memories(self, messages: list[Message]) -> None:
        """
        添加多条持久化记忆
        默认的批量处理，低效，建议重写
        :param messages: 消息列表
        :return: 无
        """
        for message in messages:
            self.add_memory(message)

    @abstractmethod
    def get_memories(self) -> List[Message]:
        """
        获取所有持久化记忆
        :return: 持久化记忆列表
        """
        pass

    @abstractmethod
    def clear_memories(self) -> None:
        """
        清空所有持久化记忆
        :return: 无
        """
        pass
