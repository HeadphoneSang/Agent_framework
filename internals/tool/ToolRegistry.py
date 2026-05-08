from typing import Dict, Any, Callable

from internals.tool import Tool
from logger.loggerUtil import get_logger


class ToolRegistry:
    """工具注册类"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}  # 复杂工具
        self._functions: dict[str, dict[str, Any]] = {}  # 简单工具，用于原型快速验证
        self.logger = get_logger()

    def register(self, tool: Tool):
        """注册工具"""
        if tool.name in self._tools:
            self.logger.warning(f"工具 {tool.name} 已经注册!将被覆盖")
        self._tools[tool.name] = tool
        self.logger.debug(f"工具 {tool.name} 注册成功")

    def unregister(self, tool: Tool):
        """注销工具"""
        if tool.name not in self._tools:
            self.logger.warning(f"工具 {tool.name} 没有注册!")
        del self._tools[tool.name]
        self.logger.debug(f"工具 {tool.name} 注销成功")

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """注册函数"""
        if name in self._functions:
            self.logger.warning(f"函数 {name} 已经注册!将被覆盖")
        self._functions[name] = {
            "description": description,
            "func": func
        }
        self.logger.debug(f"函数 {name} 注册成功")

    def get_all_tools_descriptions(self) -> str:
        """获取所有工具描述"""
        description_list = []
        # 获取所有复杂工具的信息
        for tool in self._tools.values():
            description_list.append(f"- {str(tool)}")
        # 获取所有简单工具的信息
        for name, func_info in self._functions.items():
            description_list.append(f"- function_name:{name} ; function_description:{func_info['description']}")
        # 整合为字符串提示词
        return "\n".join(description_list) if description_list else "没有注册的工具"

    def is_empty(self) -> bool:
        """判断是否为空"""
        return len(self._tools) == 0 and len(self._functions) == 0

    def execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """
        执行工具调用
        """
        tool_name = tool_call['tool_name']
        self.logger.debug(f"正在执行工具 {tool_name}...")
        if tool_name in self._tools:
            return self._tools[tool_name].run(tool_call['tool_params'])
        elif tool_name in self._functions:
            return self._functions[tool_name]['func'](**tool_call['tool_params'])
        else:
            self.logger.warning(f"工具 {tool_name} 没有注册!")
            return f"工具 {tool_name} 没有注册!"

    def get_tool_by_name(self, tool_name: str) -> Tool:
        """
        根据工具名称获取工具
        """
        if tool_name in self._tools:
            return self._tools[tool_name]
        else:
            self.logger.warning(f"工具 {tool_name} 没有注册!")
            return None
