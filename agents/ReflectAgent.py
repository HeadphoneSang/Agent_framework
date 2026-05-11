import os
import re
from typing import Optional, Dict, Any

from agents import ReactAgent
from agents.base import ToolBaseAgent, BaseMemoryAgent
from config import BaseConfig
from internals import HelloAgentsLLM, Message, BaseMemory, MemoryInMemory
from internals.tool import ToolRegistry
from logger.loggerUtil import get_logger
from utils.fileUtils import load_file_content, get_abs_path


def _extract_pure_content(message: Message):
    match = re.search(r'\[(.*?)\]:\s*(.*)', message.content, re.DOTALL)
    return match.group(2).strip() if match else message.content


class ReflectAgent(ToolBaseAgent, BaseMemoryAgent):
    def _init_memory(self, config: Dict[str, Any]) -> BaseMemory:
        return MemoryInMemory(config)

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

    def run(self, input_msg: Message, stream: bool = False, **kwargs) -> Message:
        """
        运行智能体
        """
        # 将用户的提问首先加入到消息记录
        self.add_history(input_msg)
        self.logger.info(f"Reflection Agent-{self.name}开始处理任务： {input_msg.content}")
        # 获得LLM的初次回答
        main_agent_result: Message = self.main_agent.run(input_msg, stream, **kwargs)
        main_agent_result.content = f"[初步回答内容]: {main_agent_result.content}"
        self.add_history(main_agent_result)
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
                final_answer = _extract_pure_content(self.history[-1])
                self.add_history(Message(content=final_answer, role="assistant"))
                return self.history[-1]
            # 格式化修改意见
            self.logger.debug(f"Reflection Agent-{self.name} 审阅意见: {reflection_result.content}")
            self.logger.debug(f"Reflection Agent-{self.name} 修正专家修正中...")
            reflection_result.content = f"[专家给出的修改意见]: {reflection_result.content}"
            # 添加历史记录
            self.add_history(reflection_result)
            self.refine_agent.add_history(reflection_result)
            # 修改专家给出修改结果
            refine_result: Message = self.refine_agent.run(input_msg, stream, **kwargs)
            refine_result.content = f"[完善后的内容]: {refine_result.content}"
            self.add_history(refine_result)
            self.reflection_agent.add_history(refine_result)
            self.logger.debug(f"Refine Agent-{self.name} 修正结果: {refine_result.content}")
        self.logger.debug(f"Reflection Agent-{self.name} 超出最大轮数，返回最终结果...")
        final_answer = _extract_pure_content(self.history[-1])
        self.add_history(Message(content=final_answer, role="assistant"))
        return self.history[-1]
