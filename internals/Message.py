from datetime import datetime
from typing import Optional, Any, Dict, Literal

from openai import BaseModel

MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """消息管理类"""
    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp",datetime.now()),
            metadata=kwargs.get("metadata", {})
        )
        self.time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": f"[{self.timestamp}]"+self.content,
        }

    def __str__(self):
        return f"[{self.time_str}][{self.role}]: {self.content} - "


