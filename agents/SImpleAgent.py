import os
import re
from abc import ABC
from typing import Optional, Dict, List, Any

from agents.base import BaseAgent
from config import BaseConfig
from internals import HelloAgentsLLM, Message
from internals.tool import ToolRegistry, Tool
from llms.MyLLM import MyLLM
from logger.loggerUtil import get_logger
from tools.local_tools import CalculateTool
from utils.fileUtils import load_file_content, get_abs_path


class SimpleAgent(BaseAgent):
    """
    最简单的智能体
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[BaseConfig] = None,
            tool_registry: Optional[ToolRegistry] = None,
            enable_tool_call: bool = True,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_call = enable_tool_call and self.tool_registry is not None
        self.logger = get_logger(self.name)
        self.logger.info(f"{name} 初始化完成，工具调用: {'启用' if self.enable_tool_call else '禁用'}")

    def run_msg(self, msg: str, max_tool_iterations: int = 5, **kwargs):
        return self.run(Message(content=msg, role="user"), max_tool_iterations=max_tool_iterations, **kwargs)

    def run(self, input_msg: Message, max_tool_iterations: int = 5, **kwargs) -> Message:
        """
        运行智能体
        """
        self.logger.info(f"{self.name} 正在处理: {input_msg.content}")
        # 加载系统提示词
        system_prompt = self._get_enhanced_system_prompt()
        messages = [{
            "role": "system",
            "content": system_prompt,
        }]
        for msg in self.history:
            messages.append(msg.to_openai_dict())
        messages.append(input_msg.to_openai_dict())
        self.history.append(input_msg)
        # 启动不同的对话逻辑
        if not self.enable_tool_call:
            response_content = self.llm.think(messages, **kwargs)
            messages.append({"role": "assistant", "content": response_content})
            self.logger.info(f"{self.name} 响应完成")
        elif self.enable_tool_call:
            self._run_react_loop(messages, input_msg, max_tool_iterations=max_tool_iterations, **kwargs)
        final_answer = messages[-1]['content']
        self.history.append(Message(content=final_answer, role="assistant"))
        return self.history[-1]

    def _run_react_loop(self, messages: List[Dict[str, str]], input_msg: Message, max_tool_iterations: int = 5,
                        **kwargs):
        """
        使用REACT模式进行工具调用
        """
        self.logger.info(f"{self.name} 正在使用REACT工具调用处理: {input_msg.content}")
        for epoch in range(max_tool_iterations):
            self.logger.info(f"{self.name} epoch: {epoch}")
            response_content = self.llm.think(messages, **kwargs)
            if response_content.startswith("[TOOL_CALL]"):
                # 智能助手调用工具
                tool_calls = self._parse_content_tool_call(response_content)
                if not tool_calls:
                    warn_msg = f"警告: 工具调用失败，请检查工具调用格式（{response_content}）"
                    messages.append({'role': 'tool_call', 'content': warn_msg})
                    self.logger.warning(warn_msg)
                else:
                    self.logger.info(f"工具调用：本次请求调用工具数量为 {len(tool_calls)}")
                    for tool_call in tool_calls:
                        self.logger.info(f"工具调用：正在调用工具 {tool_call.get('tool_name')}")
                        tool_result = self._execute_tool(tool_call)
                        self.logger.info(f"工具调用：工具 {tool_call.get('tool_name')} 调用完成,结果为 {tool_result}")
                        messages.append({
                            'role': 'system',
                            'content': f"调用工具 {tool_call.get('tool_name')}，结果为：{tool_result}"
                        })
                messages.append({'role': 'system', 'content': '请根据已有的历史消息，回答用户的问题'})
            else:
                # 智能助手返回结果
                messages.append({'role': 'assistant', 'content': response_content})
                self.logger.info(f"{self.name} REACT工具调用模式，响应完成")
                return
        # 超出最大相应次数
        self.logger.warning(f"超出最大响应次数，请根据已有的内容，直接回答用户问题")
        messages.append({'role': 'user', 'content': '超出最大响应次数，请根据已有的内容，直接回答用户的问题'})
        final_answer = self.llm.think(messages, **kwargs)
        messages.append({'role': 'assistant', 'content': final_answer})

    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """
        执行工具调用
        """
        result = self.tool_registry.execute_tool(tool_call)
        return result

    def _parse_content_tool_call(self, content: str) -> List[Dict[str, Any]]:
        """
        解析内容中的工具调用
        """
        ans = []
        tool_line_iterator = re.finditer(r'\[TOOL_CALL\](\w+?)\((.*?)\)', content)
        if not tool_line_iterator:
            self.logger.warning(f"未找到工具调用: {content}")
            return []
        for tool_line_match in tool_line_iterator:
            tool_call = {}
            tool_name = tool_line_match.group(1)
            tool_args_content = tool_line_match.group(2)
            #解析工具参数
            params = {}
            for param_match in re.finditer(r'(\w+?)="(.*?)"(?=,|\Z)', tool_args_content):
                param_name = param_match.group(1)
                param_value = param_match.group(2)
                params[param_name] = param_value
            tool_call['tool_name'] = tool_name
            tool_call['tool_params'] = params
            ans.append(tool_call)
        return ans

    def _get_enhanced_system_prompt(self) -> str:
        """
        获取增强后的系统提示
        """
        base_prompt = self.system_prompt or "你是一个人工智能助手，可以使用工具回答用户的问题"
        if not self.enable_tool_call or self.tool_registry is None:
            return base_prompt

        # 获取工具描述
        tools_descriptions = self.tool_registry.get_all_tools_descriptions()
        if not tools_descriptions or self.tool_registry.is_empty():
            return base_prompt
        # 获得工具提示词模板
        tool_prompt_template = load_file_content(get_abs_path(os.path.join('agents', 'prompts', 'tool_prompt.md')))
        if tool_prompt_template == "":
            raise Exception("工具提示词模板加载失败")
        tool_prompt = tool_prompt_template.format(tools_description=tools_descriptions)
        return f"{base_prompt}\n{tool_prompt}"

    def add_tool(self, tool: Tool):
        """
        添加工具
        """
        if not self.tool_registry:
            self.tool_registry = ToolRegistry()
            self.enable_tool_call = True
        self.tool_registry.register(tool)

    def remove_tool(self, tool: Tool):
        """
        移除工具
        """
        if not self.tool_registry:
            return
        self.tool_registry.unregister(tool)
        if self.tool_registry.is_empty():
            self.enable_tool_call = False

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
        if self.tool_registry.is_empty():
            self.enable_tool_call = False

    def list_tools(self):
        """
        列出所有工具
        """
        if not self.tool_registry:
            return []
        return self.tool_registry.get_all_tools_descriptions()


if __name__ == '__main__':
    llm = MyLLM()
    agent = SimpleAgent("测试助手", llm)
    agent.add_tool(CalculateTool())
    agent.run_msg("请帮我查看一下北京的天气")
    for message in agent.get_history():
        print(message)
