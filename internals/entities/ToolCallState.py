import json
from typing import Any, Dict
from dataclasses import dataclass
from .BaseState import BaseState, StateCode


@dataclass
class ToolCallState(BaseState):
    """
    工具调用状态类，继承自 BaseState
    用于记录工具调用的完整生命周期信息，包括调用参数和执行结果
    """
    tool_call_id: str
    tool_name: str
    tool_params: Dict[str, Any]
    result: Any = None

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_params: Dict[str, Any],
        result: Any = None
    ):
        """
        初始化工具调用状态

        Args:
            tool_call_id: 工具调用的唯一标识符
            tool_name: 工具名称
            tool_params: 工具调用参数
            result: 工具执行结果（默认为 None，执行后填充）
        """
        super().__init__(StateCode.TOOL_CALL, None)
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.tool_params = tool_params
        self.result = result

    def set_result(self, result: Any) -> None:
        """
        设置工具执行结果

        Args:
            result: 工具执行结果
        """
        self.result = result
        self.payload = result  # 同步更新 BaseState 的 payload

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            包含所有字段的字典
        """
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "result": self.result
        }
    def to_tool_call_message(self) -> Dict[str, Any]:
        """
        转换为 OpenAI API 格式的 tool call 消息
        用于构建消息链
        Returns:
            OpenAI 格式的 tool call 消息字典
        """
        return {
            "id": self.tool_call_id,
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(self.tool_params)
            }
        }

    @staticmethod
    def build_assistant_tool_call_message(tool_call_states: list["ToolCallState"]) -> Dict[str, Any]:
        """
        构建 OpenAI API 格式的 assistant tool_calls 消息
        OpenAI 要求：tool role 消息前必须有带 tool_calls 的 assistant 消息

        Args:
            tool_call_states: 工具调用状态列表

        Returns:
            OpenAI 格式的 assistant 消息字典，包含 tool_calls 数组
        """
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [tc.to_tool_call_message() for tc in tool_call_states]
        }

    def to_openai_message(self) -> Dict[str, Any]:
        """
        转换为 OpenAI API 格式的 tool message
        用于构建消息链

        Returns:
            OpenAI 格式的 tool 执行结果的消息字典
        """
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": str(self.result) if self.result is not None else ""
        }

    def __str__(self) -> str:
        """
        字符串表示

        Returns:
            格式化的字符串
        """
        return (
            f"ToolCallState("
            f"id={self.tool_call_id}, "
            f"name={self.tool_name}, "
            f"params={self.tool_params}, "
            f"result={self.result}"
            f")"
        )