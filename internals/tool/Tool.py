from abc import ABC, abstractmethod
from typing import Dict, Any, List

from openai import BaseModel


class ToolParameter(BaseModel):
    """
    工具参数定义
    """
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None


class Tool(ABC):
    """工具"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, params: Dict[str, Any]) -> str:
        """运行工具"""
        pass

    @abstractmethod
    def get_params(self) -> List[ToolParameter]:
        """获取工具参数"""
        pass

    def __str__(self):
        return f"tool_name: {self.name} ; tool_description: {self.description}"

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        获得openai原生工具调用的函数签名
        """
        params = self.get_params()
        properties: dict[str, Any] = {}
        required = []
        for param in params:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                prop["description"] = f"{[param.description]} (默认值: {param.default})"
            if param.type.lower() == "array":
                prop["items"] = {"types": "string"}

            properties[param.name] = prop
            # 如果参数必备，添加到required
            if param.required:
                required.append(param.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
