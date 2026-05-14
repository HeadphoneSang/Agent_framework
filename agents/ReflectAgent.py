import os
import re
from typing import Optional, Dict, Any, Iterator

from agents import ReactAgent
from agents.base import ToolBaseAgent, BaseMemoryAgent
from config import BaseConfig
from externals.memory.RedisMemory import RedisMemory
from internals import HelloAgentsLLM, Message
from internals.memory import BaseMemory
from internals.tool import ToolRegistry
from logger.loggerUtil import get_logger
from utils.fileUtils import load_file_content, get_abs_path


def _extract_pure_content(message: Message):
    match = re.search(r'\[(.*?)\]:\s*(.*)', message.content, re.DOTALL)
    return match.group(2).strip() if match else message.content


class ReflectAgent(ToolBaseAgent,BaseMemoryAgent):

    def _init_memory(self, config: Dict[str, Any]) -> BaseMemory:
        return RedisMemory(config)

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            initial_prompt: Optional[str] = None,
            reflect_prompt: Optional[str] = None,
            refine_prompt: Optional[str] = None,
            config: Optional[BaseConfig] = None,
            tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(name, llm, initial_prompt, config, tool_registry)
        self.name = name
        self._init_prompt(initial_prompt, reflect_prompt, refine_prompt)
        # 用来初步回答问题的智能体
        self.main_agent = ReactAgent(f"({name})-(回答智能体)", llm, self.initial_prompt, config, tool_registry)
        # 用来审查问题的智能体
        self.reflection_agent = ReactAgent(f"({name})-(审查智能体)", llm, self.reflect_prompt, config, tool_registry)
        # 用来修正问题的智能体
        self.refine_agent = ReactAgent(f"({name})-(修正智能体)", llm, self.refine_prompt, config, tool_registry)
        self.logger = get_logger(name)
        self.max_reflection_epoch = config.get("max_reflection_epoch", 5)

    def _generator_short_term_memories(self, session_memories: list[dict[str, str]]) -> str:
        return "- " + "\n- ".join([f"[{record['role']}]:{record['content']}" for record in session_memories])

    def _init_prompt(self, initial_prompt: str, reflect_prompt: str, refine_prompt: str):
        """
        初始化prompt, 导入default模板
        """
        if initial_prompt is None or initial_prompt == "":
            self.initial_prompt = load_file_content(get_abs_path(os.path.join('agents', 'prompts', 'react_prompt.md')))
        if reflect_prompt is None or reflect_prompt == "":
            self.reflect_prompt = load_file_content(get_abs_path(os.path.join('agents', 'prompts', 'reflection_prompt'
                                                                                                   '.md')))
        if refine_prompt is None or refine_prompt == "":
            self.refine_prompt = load_file_content(get_abs_path(os.path.join('agents', 'prompts', 'refine_prompt.md')))

    def stream(self, input_params: Dict[str, Any], question_key: str = "question", tool_key: str = "tools",
               stream: bool = False, **kwargs) -> Iterator[Message]:
        user_question = input_params[question_key]
        session_memories: list[dict[str, str]] = []
        self.logger.info(f"Reflection Agent-{self.name}开始处理任务： {user_question}")
        # 获得LLM的初次回答
        main_agent_result: Message = self.main_agent.invoke(input_params=input_params, tools_key=tool_key,
                                                            question_key=question_key, track_msg=False,
                                                            stream=stream, **kwargs)[-1]
        main_agent_result.content = f"[初步回答内容]: {main_agent_result.content}"
        # 将初步的回答加入到短期会话记忆
        session_memories.append(main_agent_result.to_openai_dict())
        yield main_agent_result
        for epoch in range(self.max_reflection_epoch):
            self.logger.debug(f"Reflection Agent-{self.name} epoch: {epoch}")
            # 获得修改意见
            self.logger.debug(f"Reflection Agent-{self.name} 审阅专家审阅中...")
            # 获取历史记录 (仅包含之前的操作记录和最近的一次修改记录)
            history_prompt = self._generator_short_term_memories(session_memories)
            reflection_result: Message = self.reflection_agent.invoke({
                "question": user_question,
                "history": history_prompt
            }, tools_key=tool_key,
                question_key=question_key, track_msg=False,
                stream=stream, **kwargs)[-1]
            if reflection_result.content.strip() == "无":
                # 返回纯净的智能体回答
                final_answer = _extract_pure_content(self.main_agent.history[-1])
                final_msg = Message(content=final_answer, role="assistant")
                session_memories.append(final_msg.to_openai_dict())
                yield final_msg
            # 格式化修改意见
            self.logger.debug(f"Reflection Agent-{self.name} 审阅意见: {reflection_result.content}")
            self.logger.debug(f"Reflection Agent-{self.name} 修正专家修正中...")
            reflection_result.content = f"[专家给出的修改意见]: {reflection_result.content}"
            # 添加历史记录到短期记忆
            session_memories.append(reflection_result.to_openai_dict())
            yield reflection_result
            # 修改专家给出修改结果
            history_prompt = self._generator_short_term_memories(session_memories)
            refine_result: Message = self.refine_agent.invoke({
                "question": user_question,
                "history": history_prompt
            }, tools_key=tool_key,
                question_key=question_key, track_msg=False,
                stream=stream, **kwargs)[-1]
            refine_result.content = f"[完善后的内容]: {refine_result.content}"
            # 将修改后的结果追加到会话短期记忆
            session_memories.append(refine_result.to_openai_dict())
            yield refine_result
            self.logger.debug(f"Refine Agent-{self.name} 修正结果: {refine_result.content}")
        self.logger.debug(f"Reflection Agent-{self.name} 超出最大轮数，返回最终结果...")
        final_answer = _extract_pure_content(Message(content=session_memories[-1]['content'], role="assistant"))
        yield Message(content=final_answer, role="assistant")

    def run(self, input_msg: Message, stream: bool = False, **kwargs) -> Message:
        """
        运行智能体
        """
        # 将用户的提问首先加入到消息记录
        self.main_agent.add_history(input_msg)
        self.logger.info(f"Reflection Agent-{self.name}开始处理任务： {input_msg.content}")
        # 获得LLM的初次回答
        main_agent_result: Message = self.main_agent.run(input_msg, stream, **kwargs)
        main_agent_result.content = f"[初步回答内容]: {main_agent_result.content}"
        # 将LLM的初次回答加入到反思专家的历史记录
        self.reflection_agent.add_history(main_agent_result)
        # 将LLM的初次回答加入到完善专家的历史记录
        self.refine_agent.add_history(main_agent_result)
        for epoch in range(self.max_reflection_epoch):
            self.logger.debug(f"Reflection Agent-{self.name} epoch: {epoch}")
            # 获得修改意见
            self.logger.debug(f"Reflection Agent-{self.name} 审阅专家审阅中...")
            reflection_result: Message = self.reflection_agent.run(input_msg, stream, **kwargs)
            if reflection_result.content.strip() == "无":
                # 返回纯净的智能体回答
                final_answer = _extract_pure_content(self.main_agent.history[-1])
                self.history.append(Message(content=final_answer, role="assistant"))
                return self.history[-1]
            # 格式化修改意见
            self.logger.debug(f"Reflection Agent-{self.name} 审阅意见: {reflection_result.content}")
            self.logger.debug(f"Reflection Agent-{self.name} 修正专家修正中...")
            reflection_result.content = f"[专家给出的修改意见]: {reflection_result.content}"
            # 添加历史记录
            self.refine_agent.add_history(reflection_result)
            # 修改专家给出修改结果
            refine_result: Message = self.refine_agent.run(input_msg, stream, **kwargs)
            refine_result.content = f"[完善后的内容]: {refine_result.content}"
            self.main_agent.add_history(refine_result)
            self.reflection_agent.add_history(refine_result)
            self.logger.debug(f"Refine Agent-{self.name} 修正结果: {refine_result.content}")
        self.logger.debug(f"Reflection Agent-{self.name} 超出最大轮数，返回最终结果...")
        final_answer = _extract_pure_content(self.main_agent.history[-1])
        self.history.append(Message(content=final_answer, role="assistant"))
        return self.history[-1]

    def clear_history(self):
        """
        清空历史
        """
        self.main_agent.clear_history()
        self.reflection_agent.clear_history()
        self.refine_agent.clear_history()
        self.history.clear()

    def add_history(self, message: Message):
        self.history.append(message)
        self.main_agent.add_history(message)
        self.reflection_agent.add_history(message)
        self.refine_agent.add_history(message)
