import os
import re
from typing import Optional, List, Dict, Any
from agents.base import BaseAgent
from agents.base.ToolBaseAgent import ToolBaseAgent
from config import BaseConfig
from internals import HelloAgentsLLM, Message
from internals.tool import ToolRegistry, Tool
from logger.loggerUtil import get_logger
from utils.fileUtils import load_file_content, get_abs_path


class ReactAgent(ToolBaseAgent):
    """
    React模式下的智能体
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[BaseConfig] = None,
            tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(name, llm, system_prompt, config, tool_registry)
        if self.system_prompt is None or self.system_prompt == "":
            self.system_prompt = load_file_content(get_abs_path(os.path.join('agents', 'prompts', 'react_prompt.md')))
        self.max_epoch = config.get("max_epoch", 5)
        self.logger = get_logger(self.name)
        self.max_history_len = config.get("max_history_length", 50)

    def run(self, input_msg: Message, stream: bool = False, **kwargs) -> Message:
        """
        运行智能体
        """
        self.logger.info(f"{self.name} 正在处理: {input_msg.content}")
        prompt_template = self.system_prompt
        tools_descriptions = self.tool_registry.get_all_tools_descriptions()
        # self.add_history(input_msg)
        # 开始执行React循环
        for epoch in range(self.max_epoch):
            self.logger.debug(f"{self.name} epoch: {epoch}")
            # 生成最近的max_history_length长度的历史记录的提示词列表
            current_history: List[str] = [str(message) for message in self.history[-self.max_history_len:]]
            history_prompt = "\n- ".join(current_history)
            prompt = prompt_template.format(history=history_prompt, question=input_msg.content,
                                            tools=tools_descriptions)
            messages = [{
                "role": "system",
                "content": prompt,
            }]
            response_content = self.llm.think(messages, **kwargs, stream=stream)
            # 解析LLM返回的格式化语句
            try:
                parts = self._b_split(response_content)
                thought_part = parts[0]
                action_part = parts[1]
                thought_content = thought_part.split("Thought:")[1].strip()
                if thought_content != "":
                    self.add_history(Message(content=thought_content, role="assistant"))
                # 解析action
                tool_call_list: List[Dict[str, Any]] = self._parse_action_part(action_part)
                # LLM没有调用工具，直接返回智能体的最终结果
                if len(tool_call_list) == 0:
                    return self.history[-1]
                # 执行所有的工具调用
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_result = self.tool_registry.execute_tool(tool_call)
                    tool_call_results.append(f"- 工具: {tool_call.get('tool_name')} 的执行结果为: {tool_result}")
                # 将所有的工具调用的结果加入到历史记录里面
                self.add_history(Message(content="\n".join(tool_call_results), role="tool"))
                self.logger.debug(f"{self.name} 工具调用结果: {self.history[-1].content}")
            except Exception as e:
                self.logger.error(f"{self.name} 运行错误: {e}")
                self.add_history(
                    Message(content=f"{response_content} **！！回答格式错误！！**，__请严格检查回答要求，并重新回复。__", role="tool"))
        # 如果超过最大轮数，则返回最终结果
        current_history: List[str] = [str(message) for message in self.history[-self.max_history_len:]]
        history_prompt = "\n- ".join(current_history)
        messages = [
            {
                "role": "system",
                "content": history_prompt
            },
            {
                "role": "system",
                "content": f"你已经超出了最大的回答次数。请结合上文的历史信息，针对用户的原始问题: {input_msg.content}\n直接给出总结性的回答文本，结束这次对话"
            }
        ]
        response_content = self.llm.think(messages, **kwargs, stream=stream)
        self.add_history(Message(content=response_content, role="assistant"))
        return self.history[-1]

    def _parse_action_part(self, action_part_content: str) -> List[Dict[str, Any]]:
        """
        解析action为执行工具的列表，每个项目表示一个工具调用。包含一个tool_name:str和一个params:str
        例子: Action: some action \n Action: some action
        :param action_part_content:
        :return:
        """
        action_iterator = re.finditer(r'Action: (.*?)(?=\s*(Action:|\Z))', action_part_content,re.DOTALL)
        if action_iterator is None:
            self.logger.error(f"{self.name} 解析Action字段错误: {action_part_content}")
            return []
        tool_call_list: List[Dict[str, Any]] = []
        for action_match in action_iterator:
            # action的具体内容：可能有: tool_name[arg0_name="arg0_value",...,argN_name="argN_value"]
            # 也有Finish[answer]
            action_content = action_match.group(1)
            if action_content.startswith("Finish"):
                # Finish[answer]
                # finish_answer = action_content.split("[")[1].split("]")[0].strip()
                # self.add_history(Message(content=finish_answer, role="assistant"))
                finish_match = re.search(r'Finish\[(.*)\]', action_content,re.DOTALL)
                finish_answer = finish_match.group(1).strip()
                self.add_history(Message(content=finish_answer, role="assistant"))
                return []
            else:
                # tool_name[args]
                try:
                    tool_match = re.search(r'(\w+)\[(.*?)\]', action_content)
                    if tool_match is None:
                        self.logger.error(f"{self.name} 解析Action字段错误: {action_content}")
                        self.add_history(
                            Message(content=f"{action_content} 错误，请检查工具调用格式，并重新回复 。", role="tool"))
                        raise Exception(f"Action字段解析错误: {action_content}")
                    tool_name = tool_match.group(1)
                    tool_args_content = tool_match.group(2)
                    tool_params = dict(re.findall(r'(\w+)="(.*?)"', tool_args_content))
                    tool_call_list.append({
                        "tool_name": tool_name,
                        "tool_params": tool_params
                    })
                except Exception as e:
                    self.logger.error(f"{self.name} 解析Action字段错误: {action_content}")
                    self.add_history(
                        Message(content=f"{action_content} 错误，请检查工具调用格式，并重新回复 。", role="tool"))
                    raise e
        return tool_call_list

    def _b_split(self, response_content) -> tuple[str, str]:
        """
        将llm回复的信息拆成Action和Thought两个字符串
        :param response_content: llm返回的相应文本
        :return: 返回一个tuple[str,str]
        """
        parts = re.split(r'(?=Action:)', response_content, maxsplit=1)
        return parts[0].strip(), parts[1].strip()
