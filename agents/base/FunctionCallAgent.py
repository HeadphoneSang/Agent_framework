import json
from abc import ABC
from typing import Any, Union, Iterator, Dict, List, Tuple, Optional
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall, ChatCompletionMessage
from pydantic import ValidationError

from agents.base import ToolBaseAgent
from agents.valid_handlers.FunctionHitValid import FunctionHitValid
from agents.valid_handlers.StructValidHandler import StructValidHandler
from config import BaseConfig
from internals import HelloAgentsLLM, Message
from internals.entities import BaseState, StateCode, ToolCallState
from agents.channels import ValidPipline
from internals.tool import ToolRegistry
from internals.tool.StructuredFunctionTool import StructuredFunctionTool
from llms.MyLLM import MyLLM
from tools.local_tools import CalculateTool, TimeTool
from tools.web_tools import SearchTool


class FunctionCallAgent(ToolBaseAgent):
    """
    功能调用代理
    基于 OpenAI 原生 function calling 机制，采用两步调用策略：
    1. 第一次调用：强制调用 final_structured_output 函数，获取 analysis（思考分析）和 need_tool（是否需要工具）
    2. 根据 need_tool 的值：
       - True:  用标准 tool_choice="auto" 让 LLM 选择并调用工具，执行工具后返回结果
       - False: 直接返回 analysis 中的最终答案
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
        self._structured_output_fn = "final_structured_output"
        self._structured_rollback_times = 3
        # 用来验证 final_structured_output 是否被调用
        self.hit_valid_pipeline = ValidPipline(
            [
                FunctionHitValid()
            ], self.logger
        )
        # 结构化输出验证
        self.struct_valid_pipeline = ValidPipline(
            [
                StructValidHandler()
            ],
            self.logger
        )

    # ────────────────────────────── 抽象方法实现 ──────────────────────────────

    def stream(self, input_params: Dict[str, Any], **kwargs) -> Iterator[BaseState]:
        """
        流式（生成器）接口，每次 yield 一个 BaseState，供 invoke() 收集。
        """
        messages = kwargs.get("messages", [{"role": "user", "content": input_params.get("question", "")}])
        for state in self.invoke_with_tools(messages, **kwargs):
            yield state

    # ────────────────────────────── Schema 构建 ──────────────────────────────

    def _build_structured_output_schema(self) -> List[Dict[str, Any]]:
        """
        构建 final_structured_output 的 tool schema，
        用于第一步强制 LLM 输出结构化分析结果。
        """
        return [StructuredFunctionTool().to_openai_schema()]

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        通过 tool_registry 构建 OpenAI 标准 function calling schema。
        :return: OpenAI tools 格式的 schema 列表
        """
        if not self.tool_registry:
            return []
        return self.tool_registry.get_tools_schema()

    # ────────────────────────────── 解析工具 ──────────────────────────────

    def _extract_message_content(self, ai_message: ChatCompletion) -> Tuple[str, str]:
        """
        从 LLM 响应中提取内容。
        优先提取 tool_calls 的 arguments；若无 tool_calls 则取文本 content。
        :param ai_message: LLM 返回的 ChatCompletion 对象
        :return: (content_or_arguments, tool_name)
                 - 有 tool_call 时: (arguments_json_str, function_name)
                 - 无 tool_call 时: (content_str, "")
        """
        choice = ai_message.choices[0].message

        if choice.tool_calls:
            tc = choice.tool_calls[0]
            return tc.function.arguments, tc.function.name

        return choice.content or "", ""

    def _parse_function_call_arguments(self, ai_msg_content: str) -> Dict[str, Any]:
        """
        解析 final_structured_output 函数返回的 JSON 字符串。
        :param ai_msg_content: LLM 返回的 arguments JSON 字符串
        :return: {"analysis": str, "need_tool": bool}
        """
        try:
            data = json.loads(ai_msg_content)
            return {
                "analysis": data.get("analysis", ""),
                "need_tool": data.get("need_tool", False)
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"解析 function call 参数失败: {e}, 原始内容: {ai_msg_content}")
            return {"analysis": ai_msg_content, "need_tool": False}

    # ────────────────────────────── 核心流程（子步骤）─────────────────────────────

    def _build_all_tools_schema(self) -> List[Dict[str, Any]]:
        """
        构建完整的工具列表：final_structured_output + 所有注册的工具。
        这样模型在第一次思考时就能看到所有可用工具。
        """
        schemas = self._build_structured_output_schema()
        schemas.extend(self._build_tool_schemas())
        return schemas

    def _inject_analysis_system_prompt(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在消息列表开头注入系统提示词，引导模型先调用 final_structured_output 进行分析。
        """
        analysis_instruction = {
            "role": "system",
            "content": (
                "## 严格指令：本次回答你必须先调用`final_structured_output`\n\n"
                "你的核心任务是根据已有信息分析用户问题。你必须严格遵守以下准则：\n\n"
                "### 第一原则（不可违反）\n"
                "- 你**只能**调用 `final_structured_output` 这一个函数，**禁止直接调用其他任何工具函数**。\n"
                "- 你必须通过 `tool_calls` 机制调用 `final_structured_output`，不要以文本形式直接输出分析内容。\n\n"
                "### 调用 final_structured_output 时的参数填写规则\n\n"
                "1. **`analysis`**（字符串，必填）\n"
                "   - 写出你的思考过程或最终答案。\n"
                "   - 如果 `need_tool` 为 false：这里应包含对用户问题的完整最终回答。\n"
                "   - 如果 `need_tool` 为 true：这里应分析需要什么信息、打算调用哪个工具来获取。\n\n"
                "2. **`need_tool`**（布尔值，必填）\n"
                "   - `false`：当前信息足够直接回答用户问题。\n"
                "   - `true`：需要调用工具获取额外信息才能回答。\n\n"
                "### 行为约束\n"
                "- 不要跳过 `final_structured_output` 直接输出文本。\n"
                "- 不要直接调用其他工具函数（如 search、calculate 等）——你只需要在 `analysis` 中说明打算用什么工具即可。\n"
                "- 即使你能直接回答某些问题，也**必须**通过 `final_structured_output` 输出。\n\n"
                "### 正确示例\n"
                "用户问：\"今天北京天气怎么样？\"\n"
                "→ 调用 final_structured_output { analysis: \"需要查询北京今天的天气信息，用户没有提供具体数据，我需要调用天气查询工具来获取。\", need_tool: true }\n\n"
                "用户问：\"2+3=?\"\n"
                "→ 调用 final_structured_output { analysis: \"2+3=5，这个问题不需要调用工具，我可以直接回答。\", need_tool: false }"
            )
        }
        return [analysis_instruction] + list(messages)

    def _valid_structured_response(self, choice: ChatCompletionMessage) -> Tuple[str, Optional[str]]:
        """
        验证并提取结构化输出函数的调用内容。
        :return: (structured_response_content, tc_id)
                 tc_id 为结构化输出函数的 tool_call_id，存在时必不为 None。
        """
        # 验证是否命中结构化输出函数
        self.hit_valid_pipeline.validate(choice, function_name=self._structured_output_fn)
        # 验证是否符合结构化输出函数的参数
        structured_tc: ChatCompletionMessageFunctionToolCall = next(
            tool for tool in choice.tool_calls if tool.function.name == self._structured_output_fn)
        structured_response_content: str = structured_tc.function.arguments  # json{}
        tc_id: str = structured_tc.id
        self.struct_valid_pipeline.validate(choice, json_content=structured_response_content,
                                            function_name=self._structured_output_fn)
        return structured_response_content, tc_id

    def _do_structured_analysis(
            self,
            messages: List[Dict[str, Any]],
            temperature: float,
            force_tool_choice: bool = True,
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Step 1: 让 LLM 输出结构化分析，同时能看到所有工具。
        :param force_tool_choice: 是否强制指定 tool_choice 为 final_structured_output。
        :return: (analysis, need_tool, tc_id)
                 tc_id 为 final_structured_output 的 tool_call_id；若模型未调用函数而是默认输出则返回 None。
        """
        self.logger.debug(f"[{self.name}] Step 1: 结构化分析...")

        # 构建完整工具列表（包含 final_structured_output + 所有注册工具）
        all_tools = self._build_all_tools_schema()

        # 注入分析提示词
        prepared_messages = self._inject_analysis_system_prompt(messages)

        # 根据 force_tool_choice 决定策略
        if force_tool_choice:
            tool_choice = {"type": "function", "function": {"name": self._structured_output_fn}}
        else:
            tool_choice = "auto"

        enable_thinking_mode = "disabled" if force_tool_choice else "enabled"

        response: ChatCompletion = self.llm.think_origin(
            prepared_messages,
            tools=all_tools,
            tool_choice=tool_choice,
            temperature=temperature,
            extra_body={"thinking": {"type": enable_thinking_mode}}  # 供调试观察 LLM 是否遵循了 tool_choice 策略
        )

        choice = response.choices[0].message
        # 三层验证，保证LLM输出稳定的结构化文本
        structured_response_content = ""
        tc_id: Optional[str] = None
        try:
            structured_response_content, tc_id = self._valid_structured_response(choice)
        except ValueError as e:
            # 让llm重新调用，自纠错
            is_correct = False
            self.logger.warning(f"初步验证结构化输出函数失败，错误原因：{str(e)}")
            correction_idx = len(prepared_messages)  # 记录纠错消息的位置，后续替换用
            rollback_msg = {"role": "system", "content": str(e)}
            prepared_messages.append(rollback_msg)
            # 进行自纠错
            for time in range(self._structured_rollback_times):
                self.logger.warning(
                    f"[{self.name}] 自纠错第 {time + 1} 次，最大尝试次数为 {self._structured_rollback_times}")
                response: ChatCompletion = self.llm.think_origin(
                    prepared_messages,
                    tools=all_tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    extra_body={"thinking": {"type": enable_thinking_mode}}  # 供调试观察 LLM 是否遵循了 tool_choice 策略
                )
                choice = response.choices[0].message
                try:
                    structured_response_content, tc_id = self._valid_structured_response(choice)
                    is_correct = True
                    break  # 自纠错成功，退出循环
                except ValueError as e:
                    self.logger.warning(f"[{self.name}] 自纠错第 {time + 1} 次失败，错误原因：{str(e)}")
                    # 替换上一条纠错消息（而非追加），防止上下文被重复的失败信息撑大
                    prepared_messages[correction_idx] = {"role": "system", "content": str(e)}
            # 自纠错失败
            if not is_correct:
                self.logger.error(f"[{self.name}] 自纠错失败，请检查模型是否正确返回结构化文本！")
                structured_response_content = json.dumps({
                    "analysis": "正在思考中....",
                    "need_tool": True
                })
                # tc_id 保持 None
        # 解析结构化文本
        parsed = self._parse_function_call_arguments(structured_response_content)
        analysis, need_tool = parsed["analysis"], parsed["need_tool"]
        return analysis, need_tool, tc_id

    def _execute_tool_calls(
            self,
            messages: List[Dict[str, Any]],
            tool_choice: Union[str, Dict],
            temperature: float,
    ) -> List[BaseState]:
        """
        Step 2a: 发起工具调用请求，执行所有工具，返回状态列表和扩展后的消息链。
        :param messages: 消息链（已包含 Step 1 的回复）
        :return: (tool_states, message_chain)
                 若 LLM 未调用任何工具，返回 ([], None)
        """
        self.logger.debug(f"[{self.name}] Step 2a: 执行工具调用...")
        tool_schemas = self._build_tool_schemas()
        # 发起新的 LLM 调用（与 Step 1 保持一致，关闭思考模式，防止 DeepSeek 等模型
        # 因 reasoning_content 缺失而报 400 错误）
        response: ChatCompletion = self.llm.think_origin(
            messages,
            tools=tool_schemas,
            tool_choice=tool_choice,
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
        choice = response.choices[0].message
        if not choice.tool_calls:
            self.logger.warning(f"[{self.name}] LLM 未调用任何工具")
            return []
        # 有工具调用，逐个工具执行
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            }
            for tc in choice.tool_calls
        ]
        # 执行所有工具调用（跳过 final_structured_output）
        tool_states: List[BaseState] = []

        # 逐一执行工具调用
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_id = tc["id"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            tool_call_dict = {"tool_name": tool_name, "tool_params": tool_args}
            try:
                tool_result = self.tool_registry.execute_tool(tool_call_dict)
                self.logger.debug(f"[{self.name}] 工具 {tool_name} 执行结果: {tool_result}")
                # 使用 ToolCallState 记录工具调用信息
                tool_state = ToolCallState(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_params=tool_args,
                    result=tool_result
                )
                tool_state.state = StateCode.SUCCESS
            except ValueError as e:
                self.logger.error(f"[{tool_name}]{e}")
                tool_state = ToolCallState(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_params=tool_args,
                    result=str(e)
                )
                tool_state.state = StateCode.FAILED
            tool_states.append(tool_state)

        # 如果没有执行任何工具，返回 None
        if not tool_states:
            self.logger.warning(f"[{self.name}] 没有实际工具被执行")
            raise RuntimeError("工具调用失败!没有实际工具被执行")
        return tool_states

    # ────────────────────────────── 核心流程（编排）─────────────────────────────

    def invoke_with_tools(
            self,
            messages: List[Dict[str, Any]],
            tool_choice: Union[str, Dict] = "auto",
            need_thought: bool = True,
            force_tool_choice: bool = True,
            **kwargs
    ) -> Iterator[BaseState]:
        """
        核心调用流程（两步策略）：

        Step 1 — 结构化分析
            模型看到所有工具（包括 final_structured_output），
            被引导先调用 final_structured_output 进行分析，输出 analysis + need_tool。
            如果模型直接调用了其他工具，则跳过分析直接进入 Step 2。

        Step 2 — 工具执行
            need_tool == True  → 执行工具调用，获取结果，发起最终请求
            need_tool == False → 直接返回 analysis 作为最终答案

        :param messages:          OpenAI 格式的消息列表
        :param tool_choice:       工具选择策略（默认 "auto"）
        :param need_thought:      是否执行 Step 1 的结构化分析；为 False 时直接跳到工具调用步骤
        :param force_tool_choice: Step 1 是否强制指定 tool_choice。
                                  某些模型（如 qwen thinking mode）不支持，需设为 False。
        :param kwargs:            额外 LLM 参数（temperature / max_tokens 等）
        :return:                  BaseState 包含 THOUGHT(思考过程) / TOOL_CALL(工具执行结果) / Finish(结束)
        """
        temperature = kwargs.get("temperature", self.temperature)
        current_messages = list(messages)

        # ── Step 1: 结构化分析（need_thought=True 时执行）────────────────
        if need_thought:
            analysis, need_tool, _tc_id = self._do_structured_analysis(
                messages, temperature, force_tool_choice
            )
            self.logger.debug(f"[{self.name}] 分析结果: need_tool={need_tool}")
            # 如果没有工具调用，则说明不需要继续调用工具，可以直接回答问题，说明当前执行连结束，返回SUCCESS
            thought_msg = Message(content=analysis, role="assistant")
            current_messages.append(thought_msg.to_openai_dict())
            structured_tc_state: ToolCallState = ToolCallState(_tc_id, self._structured_output_fn, {
                "analysis": analysis,
                "need_tool": need_tool
            }, result=analysis)
            if not need_tool:
                self.logger.debug(f"[{self.name}] 无需工具调用，直接返回结果")
                yield BaseState(StateCode.Finish, structured_tc_state)
                return
            else:
                self.logger.debug(f"[{self.name}] 需要工具调用，进入工具调用流程")
                yield BaseState(StateCode.THOUGHT, structured_tc_state)
        # ── Step 2: 工具执行流程 ────────────────────────────────────────────
        tool_states = self._execute_tool_calls(
            current_messages, tool_choice, temperature
        )

        # ── Step 3: 检查工具执行结果 ──────────────────────────────────────
        if not tool_states:
            # _execute_tool_calls 返回空列表意味着 LLM 未调用任何工具。
            # 此时不应抛出空的 TOOL_CALL，否则上游会构造 tool_calls: [] 导致 API 400 错误。
            self.logger.warning(
                f"[{self.name}] Step 1 分析需要工具，但 Step 2 LLM 实际未产生工具调用，直接返回"
            )
            fallback = analysis if need_thought and analysis else "处理完成"
            yield BaseState(StateCode.Finish, fallback)
            return

        # ── Step 4: 携带工具结果返回 ──────────────────────────────────────
        yield BaseState(StateCode.TOOL_CALL, tool_states)


# ────────────────────────────────── 测试入口 ──────────────────────────────────

if __name__ == "__main__":
    tool_register = ToolRegistry()
    tool_register.register(CalculateTool())
    tool_register.register(TimeTool())
    tool_register.register(SearchTool())

    llm = MyLLM(provider='deepseek', print_content=True)
    agent = FunctionCallAgent("test", llm, tool_registry=tool_register)

    states = agent.invoke_with_tools(
        [{"role": "user", "content": "你帮我查询一下今天济南的天气"}],
        force_tool_choice=False  # qwen thinking mode 不支持强制 tool_choice
    )
    for s in states:
        print(f"[StateCode={s.state}] {s.payload}")
