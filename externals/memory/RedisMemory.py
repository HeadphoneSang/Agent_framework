import json
from typing import List, Any, Dict

import externals.storage.RedisStorage as redis_storage
from config import RedisConfig, BaseConfig
from internals import Message
from internals.memory import BaseMemory


class RedisMemory(BaseMemory):
    """
    RedisMemory类继承自BaseMemory类，用于将消息保存到Redis数据库中。
    """

    def __init__(self, session_config: Dict[str, Any], **kwargs):
        """
        初始化RedisMemory类。
        """
        super().__init__(config=session_config, **kwargs)
        self.session_id = self.config.get("session_id")
        self.memory_key = self.config.get("memory_key")
        self.redis_config: BaseConfig = RedisConfig()
        self.memory_key = self.redis_config.get("memory_key", "memory")
        self.key = f"{self.memory_key}:{self.session_id}"

    def add_memory(self, message: Message) -> None:
        redis_storage.push_to_list_right(self.key, [message.to_openai_dict()])

    def add_memories(self, messages: list[Message]) -> None:
        redis_storage.push_to_list_right(self.key, [message.to_openai_dict() for message in messages])

    def get_memories(self) -> List[Message]:
        json_objs: List[dict] = [json.loads(obj_str) for obj_str in redis_storage.get_all_list(self.key)]
        return [Message(**json_obj) for json_obj in json_objs]

    def clear_memories(self) -> None:
        return redis_storage.rm_key(self.key)
