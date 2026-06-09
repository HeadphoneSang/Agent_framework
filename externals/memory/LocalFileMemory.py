import json
import os
from typing import List, Any, Dict

from internals import Message
from internals.memory import BaseMemory


class LocalFileMemory(BaseMemory):
    """
    本地文件记忆存储，将消息保存为 JSON Lines 格式（.jsonl）文件。
    每个 session 对应一个文件，按行追加，兼顾性能和可读性。

    文件路径：{project_root}/.memory/{session_id}.jsonl
    """

    def __init__(self, session_config: Dict[str, Any], **kwargs):
        """
        初始化本地文件记忆存储

        Args:
            session_config: 会话配置，需包含 "session_id"（默认 "default"）
        """
        super().__init__(config=session_config, **kwargs)
        self.session_id = self.config.get("session_id", "default")

        # 存储目录：项目根目录下 .memory/
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.memory_dir = os.path.join(base_dir, ".memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.file_path = os.path.join(self.memory_dir, f"{self.session_id}.jsonl")

    # ────────────────────────────── 增 ──────────────────────────────

    def add_memory(self, message: Message) -> None:
        """追加单条记忆到文件末尾"""
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(message.to_openai_dict(), ensure_ascii=False) + "\n")

    def add_memories(self, messages: list[Message]) -> None:
        """批量追加多条记忆到文件末尾"""
        with open(self.file_path, "a", encoding="utf-8") as f:
            for message in messages:
                f.write(json.dumps(message.to_openai_dict(), ensure_ascii=False) + "\n")

    # ────────────────────────────── 查 ──────────────────────────────

    def get_memories(self) -> List[Message]:
        """读取所有记忆（保持写入顺序）"""
        if not os.path.exists(self.file_path):
            return []
        messages = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                messages.append(Message(**data))
        return messages

    # ────────────────────────────── 删 ──────────────────────────────

    def clear_memories(self) -> None:
        """删除整个会话的记忆文件"""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
