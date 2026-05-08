from datetime import datetime
from typing import Dict, Any, List

from internals.tool import Tool
from internals.tool.Tool import ToolParameter
from .my_calculate_tool import my_calculate_tool


class CalculateTool(Tool):
    """数学计算工具"""

    def get_params(self) -> List[ToolParameter]:
        pass

    def __init__(self):
        super().__init__(name="local_calculate_tool",
                         description="数学计算工具,输入：(expression=python计算表达式的字符串) 输出计算结果")

    def run(self, params: Dict[str, Any]) -> str:
        return my_calculate_tool(**params)


class TimeTool(Tool):
    """时间工具"""

    def get_params(self) -> List[ToolParameter]:
        pass

    def __init__(self):
        super().__init__(name="local_time_tool",
                         description="获取当前时间,无参数")

    def run(self, params: Dict[str, Any]) -> str:
        return "当前时间是：" + str(datetime.now())
