from typing import List, Dict, Any

from pydantic import BaseModel, Field

from internals.tool import Tool
from internals.tool.Tool import ToolParameter


class StructuredFunctionTool(Tool):
    """
    结构化输出工具，用于输出结构化的回答，必须包含 analysis 和 need_tool。当need_tool为False时，analysis中应包含最终答案；当need_tool为True时，analysis中应包含调用工具的思路和分析。
    """

    def get_params(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="analysis",
                description="思考过程，比如对于当前问题的回答或者是解决问题的思路。该分析可以直接根据已有信息回答问题，也可以为下文的具体工具调用提供调用的信息。",
                required=True,
                type="string"
            ),
            ToolParameter(
                name="need_tool",
                description="是否需要调用工具",
                required=True,
                type="boolean"
            ),
        ]

    def __init__(self):
        super().__init__(name="final_structured_output",
                         description="用于输出结构化的回答，必须包含 analysis 和 need_tool。当need_tool为False时，analysis中应包含最终答案；当need_tool为True时，analysis中应包含调用工具的思路和分析。")

    def run(self, params: Dict[str, Any]) -> str:
        pass