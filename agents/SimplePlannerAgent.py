import ast
import os
from typing import Optional, Dict, List

from agents import ReactAgent
from agents.base import BaseAgent, ToolBaseAgent
from config import BaseConfig
from internals import Message, HelloAgentsLLM
from internals.tool import ToolRegistry
from utils.fileUtils import load_file_content, get_abs_path


class SimplePlannerAgent(ToolBaseAgent):
    """
    SimplePlannerAgent类继承自BaseAgent类，用于实现简单的规划型智能体。
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[Dict[str, str]] = None,
            config: Optional[BaseConfig] = None,
            tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(name, llm, system_prompt=system_prompt, config=config,
                         tool_registry=tool_registry)
        if self.system_prompt is None:
            self._init_prompt()
        self.planner_prompt_template = self.system_prompt.get("planner")
        self.executor_prompt_template = self.system_prompt.get("executor")
        self.plan_llm = self.llm
        self.executor = ReactAgent("执行者", self.llm, self.executor_prompt_template, config, tool_registry)

    def _init_prompt(self):
        planner_prompt_template = load_file_content(get_abs_path(os.path.join('agents', 'prompts', 'plan_prompt.md')))
        executor_prompt_template = load_file_content(
            get_abs_path(os.path.join('agents', 'prompts', 'executor_prompt.md')))
        self.system_prompt = {
            "planner": planner_prompt_template,
            "executor": executor_prompt_template
        }

    def run(self, input_msg: Message, temperature: float = 0.7, stream: bool = False, **kwargs) -> Message:
        # 构建历史记录提示词部分
        last_memories_prompt: str = "\n\n".join([str(msg) for msg in self.get_history()])
        tools_prompt: str = self.tool_registry.get_all_tools_descriptions()
        planner_prompt = self.planner_prompt_template.format(
            history=last_memories_prompt,
            tools=tools_prompt,
            question=input_msg.content
        )
        self.logger.info(f"Planner Agent-{self.name} 正在规划任务...")
        messages = [{
            "role": "user",
            "content": planner_prompt,
        }]
        plan_result = self.llm.think(messages, **kwargs, stream=stream)
        try:
            # 找到```python和```之间的内容
            plan_str = plan_result.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan: List[str] = ast.literal_eval(plan_str)
            for step in plan:
                self.logger.info(f"Planner Agent-{self.name} 正在执行步骤: {step}")
                self.executor.run(Message(role="user", content=step), temperature=temperature, stream=stream)
        except (ValueError, SyntaxError, IndexError) as e:
            self.logger.error(f"解析计划时出错: {e}")
            self.logger.error(f"原始响应: {plan_result}")
            raise e
        except Exception as e:
            self.logger.error(f"解析计划时发生未知错误: {e}")
            self.logger.error(f"原始响应: {plan_result}")
            raise e
