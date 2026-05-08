import os
import re
from abc import ABC
from typing import Optional, List, Dict, Any
from agents.base import BaseAgent
from config import BaseConfig
from internals import HelloAgentsLLM, Message
from internals.tool import ToolRegistry, Tool
from logger.loggerUtil import get_logger
from utils.fileUtils import load_file_content, get_abs_path


class ToolBaseAgent(BaseAgent, ABC):

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[BaseConfig] = None,
            tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry

    def add_tool(self, tool: Tool):
        """
        添加工具
        """
        if not self.tool_registry:
            self.tool_registry = ToolRegistry()
        self.tool_registry.register(tool)

    def remove_tool(self, tool: Tool):
        """
        移除工具
        """
        if not self.tool_registry:
            return
        self.tool_registry.unregister(tool)

    def has_tool(self):
        return self.tool_registry is not None

    def remove_tool_by_name(self, name: str):
        """
        移除工具
        """
        if not self.tool_registry:
            return
        tool = self.tool_registry.get_tool_by_name(name)
        if not tool:
            self.logger.info(f"工具 {name} 不存在")
            return
        self.tool_registry.unregister(tool)

    def list_tools(self):
        """
        列出所有工具
        """
        if not self.tool_registry:
            return []
        return self.tool_registry.get_all_tools_descriptions()
