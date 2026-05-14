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


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


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

    def _inject_tool_prompt(self, tool_key: str):
        """
        将工具提示词注入到系统提示词里面
        :param tool_key: 工具提示词的key
        :return: None
        """
        tools_descriptions = self.tool_registry.get_all_tools_descriptions()
        kv_dict = SafeDict()
        kv_dict[tool_key] = tools_descriptions
        self.system_prompt = self.system_prompt.format_map(kv_dict)

    def _run_with_tools(self, messages: list[dict[str, str]],
                        tools_key: str = "tools", temperature: float = None,
                        stream: bool = None):
        """
        自动在提示词内注入工具信息，同时将其他的kv也注入到提示词，然后调用LLM返回文本
        :param messages: 发送的消息
        :param tools_key: 工具提示词的key
        :param temperature: 温度系数
        :param stream: 是否流式返回
        :return: 返回str
        """
        # 注入工具信息到系统提示词
        self._inject_tool_prompt(tools_key)
        response_text = self.llm.think(messages, temperature, stream)
        return response_text
