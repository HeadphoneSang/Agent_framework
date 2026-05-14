import os
import re
from typing import Optional, List, Dict, Any, Iterator
from agents.base.ToolBaseAgent import ToolBaseAgent, SafeDict
from config import BaseConfig, AgentConfig
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
            config: Optional[BaseConfig] = AgentConfig(),
            tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(name, llm, system_prompt, config, tool_registry)
        if self.system_prompt is None or self.system_prompt == "":
            self.system_prompt = load_file_content(get_abs_path(os.path.join('agents', 'prompts', 'react_prompt.md')))
        self.max_epoch = config.get("max_epoch", 5)
        self.logger = get_logger(self.name)
        self.max_history_len = config.get("max_history_length", 50)

    def _format_messages(self, input_params: Dict[str, Any], session_memories: list[dict[str, str]]):
        messages = [
            {
                'role': 'user',
                'content': self.system_prompt.format_map(SafeDict(**input_params))
            },
            {
                'role': 'system',
                'content': '## 下文是当前会话的短期聊天记录'
            },
            *session_memories
        ]
        return messages

    def _execute_tool_calls(self, tool_call_list) -> Message:
        tool_call_results = []
        for tool_call in tool_call_list:
            tool_result = self.tool_registry.execute_tool(tool_call)
            tool_call_results.append(f"- 工具: {tool_call.get('tool_name')} 的执行结果为: {tool_result}")
        # 将所有的工具调用的结果返回
        tools_results = Message(content="\n".join(tool_call_results), role="user")
        return tools_results

    def stream(self, input_params: Dict[str, Any], tools_key: str = "tools",
               question_key: str = "question",
               stream: bool = False,
               **kwargs) -> Iterator[Message]:
        """
                执行基于 ReAct 循环的智能体推理流程，支持工具调用与流式输出。
                :param input_params: 输入参数字典
                :param tools_key: 工具字段 key（默认 "tools"）
                :param question_key: 用户问题字段 key（默认 "question"）
                :param stream: 是否启用流式输出
                :param kwargs: 其他 LLM 参数
                :return: 迭代器（track_msg=True）或最终 Message
                """
        user_question = input_params[question_key]
        session_memories: list[dict[str, str]] = []
        self.logger.info(f"{self.name} 正在处理: {user_question}")
        # 开始执行React循环
        self._inject_tool_prompt(tool_key="tools")
        for epoch in range(self.max_epoch):
            self.logger.debug(f"{self.name} epoch: {epoch}")
            # 将当前的短期聊天记录注入到messages里面，然后执行工具会话调用函数
            messages = self._format_messages(input_params, session_memories)
            response_content = self.llm.think(messages, self.temperature, stream)
            # 解析LLM返回的格式化语句
            try:
                parts = self._b_split(response_content)
                thought_part = parts[0]
                action_part = parts[1]
                thought_content = thought_part.split("Thought:")[1].strip()
                # 将模型执行的思路返回
                if thought_content != "":
                    thought_msg = Message(content=thought_content, role="assistant")
                    session_memories.append(thought_msg.to_openai_dict())
                    yield thought_msg
                # 解析action
                action_state: Dict[str, Any] = self._parse_action_part(action_part)
                # LLM没有调用工具，直接返回智能体的最终结果
                if action_state['status'] == 'Finish':
                    session_memories.append(action_state["payload"].to_openai_dict())
                    yield action_state['payload']
                    return
                # 执行所有的工具调用, 并将执行结果加入会话记录，同时返回调用结果
                tool_call_list = action_state['payload']
                tools_results = self._execute_tool_calls(tool_call_list)
                yield tools_results
                session_memories.append(tools_results.to_openai_dict())
                self.logger.debug(f"{self.name} 工具调用结果: {tools_results}")
            except Exception as e:
                self.logger.error(f"{self.name} 运行错误: {e}")
                # 将格式解析错误的信息加入到短期记忆，让LLM知道错误，并重新执行
                error_msg = Message(
                    content=f"{response_content} **！！回答格式错误！！**，__请严格检查回答要求，并重新回复。__",
                    role="tool")
                session_memories.append(error_msg.to_openai_dict())
                yield error_msg
        # 如果超过最大轮数，则返回最终结果
        messages = [
            *self._format_messages(input_params, session_memories),
            {
                "role": "user",
                "content": f"你已经超出了最大的回答次数。请结合上文的历史信息，针对用户的原始问题: {user_question}\n直接给出总结性的回答文本，结束这次对话"
            }
        ]
        self.logger.warning(f"{self.name} 运行超过最大轮数，返回最终结果...")
        response_content = self.llm.think(messages, **kwargs, stream=stream)
        session_memories.append(Message(role="assistant", content=response_content).to_openai_dict())
        yield Message(content=response_content, role="assistant")

    def run(self, input_msg: Message, stream: bool = False, **kwargs) -> Message:
        """
        运行智能体
        """
        self.logger.info(f"{self.name} 正在处理: {input_msg.content}")
        prompt_template = self.system_prompt
        tools_descriptions = self.tool_registry.get_all_tools_descriptions()
        # 初始化当前会话的短期记忆列表
        short_term_memory: List[str] = [str(message) for message in
                                        self.history[-(self.max_history_len - self.max_epoch):]]
        # 开始执行React循环
        for epoch in range(self.max_epoch):
            self.logger.debug(f"{self.name} epoch: {epoch}")
            history_prompt = "\n- ".join(short_term_memory)
            prompt = prompt_template.format(history=history_prompt, question=input_msg.content,
                                            tools=tools_descriptions)
            messages = [{
                "role": "user",
                "content": prompt,
            }]
            response_content = self.llm.think(messages, **kwargs, stream=stream)
            # 解析LLM返回的格式化语句
            try:
                parts = self._b_split(response_content)
                thought_part = parts[0]
                action_part = parts[1]
                thought_content = thought_part.split("Thought:")[1].strip()
                # 将模型执行的思路加入到短期记忆记录里面
                if thought_content != "":
                    short_term_memory.append(str(Message(content=thought_content, role="assistant")))
                # 解析action
                action_state: Dict[str, Any] = self._parse_action_part(action_part)
                if action_state['status'] == 'Finish':
                    self.add_history(action_state['payload'])
                    return self.history[-1]
                # 执行所有的工具调用, 并将执行结果加入会话记录，同时返回调用结果
                tool_call_list = action_state['payload']
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_result = self.tool_registry.execute_tool(tool_call)
                    tool_call_results.append(f"- 工具: {tool_call.get('tool_name')} 的执行结果为: {tool_result}")
                # 将所有的工具调用的结果加入到短期记忆记录里面
                short_term_memory.append(str(Message(content="\n".join(tool_call_results), role="tool")))
                # self.add_history(Message(content="\n".join(tool_call_results), role="tool"))
                self.logger.debug(f"{self.name} 工具调用结果: {short_term_memory[-1]}")
            except Exception as e:
                self.logger.error(f"{self.name} 运行错误: {e}")
                # 将格式解析错误的信息加入到短期记忆，让LLM知道错误，并重新执行
                short_term_memory.append(
                    str(Message(content=f"{response_content} **！！回答格式错误！！**，__请严格检查回答要求，并重新回复。__",
                                role="tool")))
        # 如果超过最大轮数，则返回最终结果
        current_history: List[str] = [str(message) for message in self.history[-self.max_history_len:]]
        history_prompt = "\n- ".join(current_history)
        messages = [
            {
                "role": "user",
                "content": history_prompt
            },
            {
                "role": "user",
                "content": f"你已经超出了最大的回答次数。请结合上文的历史信息，针对用户的原始问题: {input_msg.content}\n直接给出总结性的回答文本，结束这次对话"
            }
        ]
        self.logger.warning(f"{self.name} 运行超过最大轮数，返回最终结果...")
        response_content = self.llm.think(messages, **kwargs, stream=stream)
        self.add_history(Message(content=response_content, role="assistant"))
        return self.history[-1]

    def _parse_action_part(self, action_part_content: str) -> Dict[str, Any]:
        """
        解析action为执行工具的列表，每个项目表示一个工具调用。包含一个tool_name:str和一个params:str
        例子: Action: some action \n Action: some action
        :param action_part_content:
        :return:
        """
        action_iterator = re.finditer(r'Action: (.*?)(?=\s*(Action:|\Z))', action_part_content, re.DOTALL)
        if action_iterator is None:
            self.logger.error(f"{self.name} 解析Action字段错误: {action_part_content}")
            raise Exception(f"{self.name} 解析Action字段错误: {action_part_content}")
        tool_call_list: List[Dict[str, Any]] = []
        for action_match in action_iterator:
            # action的具体内容：可能有: tool_name[arg0_name="arg0_value",...,argN_name="argN_value"]
            # 也有Finish[answer]
            action_content = action_match.group(1)
            if action_content.startswith("Finish"):
                finish_match = re.search(r'Finish\[(.*)\]', action_content, re.DOTALL)
                finish_answer = finish_match.group(1).strip()
                return {
                    "status": "Finish",
                    "payload": Message(content=finish_answer, role="assistant")
                }
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
        return {
            "status": "tool_calls",
            "payload": tool_call_list
        }

    def _b_split(self, response_content) -> tuple[str, str]:
        """
        将llm回复的信息拆成Action和Thought两个字符串
        :param response_content: llm返回的相应文本
        :return: 返回一个tuple[str,str]
        """
        parts = re.split(r'(?=Action:)', response_content, maxsplit=1)
        return parts[0].strip(), parts[1].strip()
