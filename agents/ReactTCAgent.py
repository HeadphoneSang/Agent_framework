import json
import os
import re
from typing import Optional, List, Dict, Any, Iterator

from openai.types.chat import ChatCompletionMessage

from agents.base.ToolBaseAgent import ToolBaseAgent, SafeDict
from agents.base.FunctionCallAgent import FunctionCallAgent
from config import BaseConfig, AgentConfig
from internals import HelloAgentsLLM, Message
from internals.entities import BaseState, ToolCallState
from internals.entities.BaseState import StateCode
from internals.tool import ToolRegistry, Tool
from logger.loggerUtil import get_logger
from utils.fileUtils import load_file_content, get_abs_path


class ReactTCAgent(FunctionCallAgent):
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
            self.system_prompt = load_file_content(
                get_abs_path(os.path.join('agents', 'prompts', 'react_tc_prompt.md')))
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

    @staticmethod
    def _extract_from_structured_fn(state: ToolCallState):
        analysis = state.tool_params['analysis']
        need_tool = state.tool_params['need_tool']
        return analysis, need_tool, state.tool_call_id

    def _record_structured_output(self, tc_state: ToolCallState, session_memories: list) -> Message:
        """记录结构化输出到短期记忆，返回 AI 消息"""
        analysis, _, tc_id = self._extract_from_structured_fn(tc_state)
        if tc_state.payload and isinstance(tc_state.payload, ChatCompletionMessage):
            # 将原始的LLM返回的消息追加到历史记录，提高缓存命中率
            session_memories.append(tc_state.payload)
        if not tc_state.payload:
            session_memories.append(tc_state.to_openai_message())
        # if tc_id:  # 将tc的请求也加到消息列表，保证过程完整性，防止模型出现幻觉
        #     tool_call_msg = ToolCallState.build_assistant_tool_call_message([tc_state])
        #     session_memories.append(tool_call_msg)
        # 返回AI的思考过程
        ai_msg = Message(role="assistant", content=analysis)
        analysis_msg_dict = ai_msg.to_openai_dict()
        if tc_id:  # 将tc的执行结果也加进去, 思考模式没有id，所以这里不执行
            analysis_msg_dict['tool_call_id'] = tc_id
            analysis_msg_dict['name'] = tc_state.tool_name
            analysis_msg_dict['role'] = 'tool'
            session_memories.append(analysis_msg_dict)
        return ai_msg

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
        # 单次会话的短期记忆
        session_memories: list[dict[str, str]] = []
        self.logger.info(f"{self.name} 正在处理: {user_question}")
        # 将工具信息注入系统提示中，供 LLM 选择调用
        self._inject_tool_prompt(tool_key="tools")
        # 开始执行React循环
        for epoch in range(self.max_epoch):
            self.logger.debug(f"{self.name} epoch: {epoch}")
            # 将当前的短期聊天记录注入到messages里面，然后执行工具会话调用函数
            messages = self._format_messages(input_params, session_memories)
            for state in self.invoke_with_tools(messages, force_tool_choice=True, **kwargs):
                if state == StateCode.THOUGHT:
                    ai_msg = self._record_structured_output(state.payload, session_memories)
                    yield ai_msg
                elif state == StateCode.TOOL_CALL:
                    tool_call_states: list[ToolCallState] = state.payload
                    # OpenAI API 要求：tool role 消息前必须有带 tool_calls 的 assistant 消息'
                    if not self.supports_thinking:
                        tool_call_msg = ToolCallState.build_assistant_tool_call_message(tool_call_states)
                        session_memories.append(tool_call_msg)
                    for tc_state in tool_call_states:
                        session_memories.append(tc_state.to_openai_message())
                        yield Message.from_open_ai(tc_state.to_openai_message())
                elif state == StateCode.Finish:
                    ai_msg = self._record_structured_output(state.payload, session_memories)
                    yield ai_msg
                    break
            else:
                # 内层循环正常结束（未收到 Finish），继续下一轮 epoch
                continue
            # 内层循环被 break（收到 Finish），结束整个 ReAct 循环
            break
        # 所有 epoch 用完仍未收到 Finish，做一次兜底总结
        if isinstance(session_memories[-1],ChatCompletionMessage) and (getattr(session_memories[-1], 'content') and getattr(session_memories[-1], 'reasoning_content')):
            yield Message(content=getattr(session_memories[-1], 'content'), role="assistant")
            print(1)
            return
        messages = [
            *self._format_messages(input_params, session_memories),
            {
                "role": "user",
                "content": f"你已经完成了所有信息的收集。请结合上文的历史信息，针对用户的原始问题: {user_question}\n直接给出总结性的回答文本，结束这次对话"
            }
        ]
        response_content = self.llm.think(messages, **kwargs, stream=stream)
        session_memories.append(Message(role="assistant", content=response_content).to_openai_dict())
        yield Message(content=response_content, role="assistant")
