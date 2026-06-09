import ast
import os
from typing import Optional, Dict, List, Any, Iterator

from agents import ReactAgent
from agents.base import BaseAgent, ToolBaseAgent
from config import BaseConfig
from internals import Message, HelloAgentsLLM
from internals.tool import ToolRegistry
from utils.fileUtils import load_file_content, get_abs_path
from utils.prompt_utils import format_messages


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

    def stream(self, input_params: Dict[str, Any], question_key: str = "question",plan_key: str = "plan", tool_key: str = "tools",
               execute_history_key: str = "execute_history", current_step_key: str = "current_step",
               stream: bool = False, **kwargs) -> Iterator[Message]:

        self.logger.info(f"Planner Agent-{self.name} 正在规划任务...")
        user_question = input_params[question_key]
        # plan的短期记忆，用来记录执行过程
        session_memories: list[Message] = []
        messages = format_messages(input_params=input_params, prompt_template=self.planner_prompt_template,session_memories=[])
        plan_result = self.llm.think(messages, **kwargs, stream=stream)
        try:
            # 找到```python和```之间的内容
            plan_str = plan_result.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan: List[str] = ast.literal_eval(plan_str)
            plan.append("总结所有的执行步骤，根据用户的问题，给出总结性的最终结果")
            # 如果解析成功，返回列表字符
            yield Message(role="assistant", content=plan_str)
            executor_params = {
                question_key: user_question,
                plan_key: plan,
                # 当前步骤和历史在执行时，动态注入
            }
            for step in plan:
                self.logger.info(f"Planner Agent-{self.name} 正在执行步骤: {step}")
                # 获取此步之前的步骤的执行历史
                cur_step_histories = "- " + "\n- ".join([str(memory) for memory in session_memories])
                # 注入执行历史
                executor_params[execute_history_key] = cur_step_histories
                # 注入当前步骤
                executor_params[current_step_key] = step
                cur_step_result = self.executor.invoke(input_params=executor_params, stream=stream, **kwargs)[-1]
                session_memories.append(cur_step_result)
                yield cur_step_result
        except (ValueError, SyntaxError, IndexError) as e:
            self.logger.error(f"解析计划时出错: {e}")
            self.logger.error(f"原始响应: {plan_result}")
            yield Message(role="assistant", content=f"解析计划时出错，请检查回复格式 {plan_result}")
            raise e
        except Exception as e:
            self.logger.error(f"解析计划时发生未知错误: {e}")
            self.logger.error(f"原始响应: {plan_result}")
            yield Message(role="assistant", content=f"解析计划时出错，请检查回复格式 {plan_result}")
            raise e

    def _init_prompt(self):
        planner_prompt_template = load_file_content(get_abs_path(os.path.join('agents', 'prompts', 'plan_prompt.md')))
        executor_prompt_template = load_file_content(
            get_abs_path(os.path.join('agents', 'prompts', 'executor_prompt.md')))
        self.system_prompt = {
            "planner": planner_prompt_template,
            "executor": executor_prompt_template
        }
