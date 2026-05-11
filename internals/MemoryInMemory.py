from typing import List, Any, Dict

from .BaseMemory import BaseMemory
from internals import Message


class MemoryInMemory(BaseMemory):
    """
    内存内存储
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.memories: List[Message] = []

    def add_memory(self, message: Message) -> None:
        self.memories.append(message)

    def get_memories(self) -> List[Message]:
        return self.memories.copy()

    def clear_memories(self) -> None:
        self.memories.clear()
