import json
import os
from typing import Any, Union, Iterator, Dict, List, Tuple, Optional
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall, ChatCompletionMessage
from agents.base import ToolBaseAgent
from agents.valid_handlers.FunctionHitValid import FunctionHitValid
from agents.valid_handlers.StructValidHandler import StructValidHandler
from config import BaseConfig
from internals import HelloAgentsLLM, Message
from internals.entities import BaseState, StateCode, ToolCallState
from agents.channels import ValidPipline
from internals.tool import ToolRegistry
from internals.tool.StructuredFunctionTool import StructuredFunctionTool
from utils.fileUtils import get_abs_path
from utils.configUtils import load_yml
from utils.dict_utils import union_dict


class FunctionCallAgent(ToolBaseAgent):
    """
    功能调用代理
    基于 OpenAI 原生 function calling 机制，采用两步调用策略：

    根据模型能力自动选择策略：
    ─ 非 thinking 模型（默认）：两步调用
        1. 第一次调用：强制调用 final_structured_output 函数，获取 analysis（思考分析）和 need_tool（是否需要工具）
        2. 根据 need_tool 的值：
           - True:  用标准 tool_choice="auto" 让 LLM 选择并调用工具，执行工具后返回结果
           - False: 直接返回 analysis 中的最终答案

    ─ thinking 模型（如 DeepSeek / Qwen thinking mode）：单次调用
        1. 一次 LLM 调用，thinking 开启，模型在内部推理中同时完成思考 + 工具决策
        2. 从 reasoning_content 提取思考过程作为 analysis
        3. yield 协议与两步策略完全一致，外层调用方无需感知差异

    模型是否支持 thinking 由 _detect_thinking_capability() 自动检测，
    检测结果缓存在 configs/thinking_wlist.yml 白名单中。
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

        # 检测当前模型是否支持 thinking 模式，并记录到实例变量
        self.supports_thinking: bool = self._detect_thinking_capability()
        self.logger.info(f"[{self.name}] 模型 thinking 能力: {self.supports_thinking}")

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

    # ────────────────────────────── Thinking 能力检测 ────────────────────────────

    def _load_thinking_wlist(self) -> dict:
        """
        加载 thinking 能力白名单。
        :return: { "provider:model_name": true/false, ... }
        """
        wlist_path = get_abs_path('configs/thinking_wlist.yml')
        if not os.path.exists(wlist_path):
            return {}
        try:
            data = load_yml(wlist_path)
            return data if data else {}
        except Exception as e:
            self.logger.warning(f"[{self.name}] 加载 thinking 白名单失败: {e}")
            return {}

    def _save_thinking_wlist(self, wlist: dict):
        """保存 thinking 能力白名单到 YAML 文件"""
        wlist_path = get_abs_path('configs/thinking_wlist.yml')
        try:
            import yaml
            with open(wlist_path, 'w', encoding='utf-8') as f:
                yaml.dump(wlist, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.logger.warning(f"[{self.name}] 保存 thinking 白名单失败: {e}")

    def _probe_thinking(self) -> bool:
        """
        发送试探性请求检测模型是否支持 thinking 模式。
        发送一条极简消息并启用 thinking，通过以下两种方式判断：
          1. API 响应中包含 reasoning_content 字段
          2. API 调用本身未抛出异常（说明接受了 thinking 参数）
        :return: True 表示模型支持 thinking 模式，False 表示不支持。
        """
        try:
            probe_messages = [{"role": "user", "content": "hi"}]
            response: ChatCompletion = self.llm.think_origin(
                probe_messages,
                temperature=0.1,
                max_tokens=5,
                extra_body={"thinking": {"type": "enabled"}}
            )
            # 检查响应中是否包含 reasoning_content（thinking 模式的标志字段）
            choice = response.choices[0]
            msg = choice.message

            # 方式1：直接属性访问（OpenAI SDK 将未知字段映射为属性）
            reasoning_content = getattr(msg, 'reasoning_content', None)
            if reasoning_content:
                return True

            # 方式2：choice 级别也可能携带该字段
            reasoning_content = getattr(choice, 'reasoning_content', None)
            if reasoning_content:
                return True

            # 方式3：通过 model_dump() 检查 Pydantic model_extra 字典
            try:
                msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else {}
                if msg_dict.get('reasoning_content') or msg_dict.get('model_extra', {}).get('reasoning_content'):
                    return True
            except Exception:
                pass

            try:
                choice_dict = choice.model_dump() if hasattr(choice, 'model_dump') else {}
                if choice_dict.get('reasoning_content') or choice_dict.get('model_extra', {}).get('reasoning_content'):
                    return True
            except Exception:
                pass

            # 调用成功但没有 reasoning_content → 不支持 thinking（或静默忽略了参数）
            return False

        except Exception as e:
            # API 拒绝了 thinking 参数 → 不支持 thinking
            self.logger.debug(f"[{self.name}] Thinking 能力试探失败（模型不支持）: {e}")
            return False

    def _detect_thinking_capability(self) -> bool:
        """
        检测当前 LLM 是否支持 thinking 模式。
        检测顺序：白名单查询 → 试探请求 → 结果持久化到白名单。
        检测结果同时保存在 self.supports_thinking 实例变量中。
        """
        provider = getattr(self.llm, 'provider', 'unknown')
        model = getattr(self.llm, 'model', 'unknown')
        model_key = f"{provider}:{model}"

        # 1. 检查白名单
        wlist = self._load_thinking_wlist()
        if model_key in wlist:
            supported = wlist[model_key]
            self.logger.info(f"[{self.name}] 白名单命中 {model_key} → thinking={supported}")
            return supported

        # 2. 白名单未命中，发送试探请求
        self.logger.info(f"[{self.name}] {model_key} 不在 thinking 白名单中，发送试探请求...")
        supported = self._probe_thinking()

        # 3. 将结果保存到白名单（供后续会话复用）
        wlist[model_key] = supported
        self._save_thinking_wlist(wlist)
        self.logger.info(f"[{self.name}] {model_key} thinking 检测完成={supported}，已写入白名单")

        return supported

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

    def _valid_structured_response(self, choice: ChatCompletionMessage, **validate_kwargs) -> Tuple[str, Optional[str]]:
        """
        验证并提取结构化输出函数的调用内容。
        :param validate_kwargs: 透传给验证管道的额外参数（如 escalation_level）。
        :return: (structured_response_content, tc_id)
                 tc_id 为结构化输出函数的 tool_call_id，存在时必不为 None。
        """
        # 验证是否命中结构化输出函数
        self.hit_valid_pipeline.validate(choice, function_name=self._structured_output_fn, **validate_kwargs)
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
            **llm_client_kwargs
    ) -> Tuple[str, bool, Optional[str], ChatCompletionMessage]:
        """
        Step 1: 让 LLM 输出结构化分析，同时能看到所有工具。
        :param force_tool_choice: 是否强制指定 tool_choice 为 final_structured_output。
        :return: (analysis, need_tool, tc_id, choice)
                 tc_id 为 final_structured_output 的 tool_call_id；若模型未调用函数而是默认输出则返回 None。
                 choice 为最后被调用的 LLM 响应的 ChatCompletionMessage 对象。
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

        client_params = union_dict(
            {
                'tools': all_tools,
                'tool_choice': tool_choice,
                'temperature': temperature,
                'extra_body': {"thinking": {"type": enable_thinking_mode}}
            }, llm_client_kwargs
        )

        response: ChatCompletion = self.llm.think_origin(
            message=prepared_messages,
            **client_params
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
                client_params = union_dict(
                    {
                        'tools': all_tools,
                        'tool_choice': tool_choice,
                        'temperature': temperature,
                        'extra_body': {"thinking": {"type": enable_thinking_mode}}
                    }, llm_client_kwargs
                )
                response: ChatCompletion = self.llm.think_origin(
                    message=prepared_messages,
                    **client_params
                )
                choice = response.choices[0].message
                try:
                    structured_response_content, tc_id = self._valid_structured_response(
                        choice, escalation_level=time + 1)
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
                choice = None
                # tc_id 保持 None
        # 解析结构化文本
        parsed = self._parse_function_call_arguments(structured_response_content)
        analysis, need_tool = parsed["analysis"], parsed["need_tool"]
        return analysis, need_tool, tc_id, choice

    # ────────────────────────────── 工具执行（公共方法）──────────────────────────

    def _execute_tool_states(
            self,
            tool_calls: List[ChatCompletionMessageFunctionToolCall],
    ) -> List[ToolCallState]:
        """
        执行 OpenAI 返回的 tool_calls，返回 ToolCallState 列表。
        抽离为公共方法，供 _execute_tool_calls 和 _thinking_invoke 共用。
        """
        tool_states: List[ToolCallState] = []
        for tc in tool_calls:
            tool_name = tc.function.name
            tool_id = tc.id
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tool_call_dict = {"tool_name": tool_name, "tool_params": tool_args}
            try:
                tool_result = self.tool_registry.execute_tool(tool_call_dict)
                self.logger.debug(f"[{self.name}] 工具 {tool_name} 执行结果: {tool_result}")
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
        return tool_states

    def _execute_tool_calls(
            self,
            messages: List[Dict[str, Any]],
            tool_choice: Union[str, Dict],
            temperature: float,
            **llm_client_kwargs
    ) -> List[ToolCallState]:
        """
        Step 2a: 发起 LLM 调用并执行所有工具。
        :return: List[ToolCallState]，若 LLM 未调用任何工具则返回空列表。
        """
        self.logger.debug(f"[{self.name}] Step 2a: 执行工具调用...")
        tool_schemas = self._build_tool_schemas()
        client_params = union_dict(
            {
                'tools': tool_schemas,
                'tool_choice': tool_choice,
                'temperature': temperature,
                'extra_body': {"thinking": {"type": "disabled"}}
            }, llm_client_kwargs
        )
        response: ChatCompletion = self.llm.think_origin(
            message=messages,
            **client_params
        )
        choice = response.choices[0].message
        if not choice.tool_calls:
            self.logger.warning(f"[{self.name}] LLM 未调用任何工具")
            return []
        return self._execute_tool_states(choice.tool_calls)

    # ────────────────────────────── Thinking 模式调用 ─────────────────────────────

    def _thinking_invoke(
            self,
            messages: List[Dict[str, Any]],
            tool_choice: Union[str, Dict],
            temperature: float,
            **llm_client_kwargs,
    ) -> Iterator[BaseState]:
        """
        Thinking 模式调用策略（单次调用）。

        适用于支持 thinking 的模型（如 deepseek-reasoner、qwen thinking mode）。
        模型在一次内部推理中同时完成思考 + 工具决策，无需两步。
        但 yield 协议与原始两步策略完全一致，外层调用方无需感知差异。

        Yield 协议：
          Finish → ToolCallState({analysis, need_tool=False})     ← 无需工具
          THOUGHT → ToolCallState({analysis, need_tool=True})     ← 思考过程
          TOOL_CALL → List[ToolCallState]                         ← 工具执行结果

        流程：
          1. 单次 LLM 调用（thinking enabled），传入所有真实工具
          2. 模型在 thinking 过程中自然决定是否调用工具
          3. 从 reasoning_content（或 content）提取思考过程作为 analysis
          4. 工具调用直接在本次响应中获取并执行
        """
        self.logger.debug(f"[{self.name}] Thinking 模式：单次调用...")

        tool_schemas = self._build_tool_schemas()

        # ── 单次 LLM 调用，启用 thinking ──
        client_params = union_dict({
            'tools': tool_schemas,
            'tool_choice': tool_choice,
            'temperature': temperature,
            'extra_body': {"thinking": {"type": "enabled"}},
        }, llm_client_kwargs)
        response: ChatCompletion = self.llm.think_origin(
            message=messages,
            **client_params
        )
        choice = response.choices[0].message
        extra_dict: dict = response.usage.model_extra or {}
        prompt_cache_hit_tokens = extra_dict.get('prompt_cache_hit_tokens',1)
        prompt_cache_miss_tokens = extra_dict.get('prompt_cache_miss_tokens',1)
        self.logger.info(f"[{self.name}] Thinking 模式：LLM 响应，tokens: {prompt_cache_hit_tokens} hit + {prompt_cache_miss_tokens} miss，缓存命中率 {(prompt_cache_hit_tokens / (prompt_cache_hit_tokens + prompt_cache_miss_tokens))*100}%")
        # ── 提取思考过程 ──
        # reasoning_content 是 thinking 模式下模型内部思考过程的文本
        reasoning = getattr(choice, 'reasoning_content', None) or ''
        content = choice.content or ''
        analysis = reasoning or content  # 优先用 reasoning_content 作为分析文本

        if not choice.tool_calls:
            # ── 无需工具，直接回答 ──
            self.logger.debug(f"[{self.name}] Thinking 模式：无需工具，直接返回结果")
            yield BaseState(StateCode.Finish, ToolCallState(
                None,
                self._structured_output_fn,
                dict(analysis=analysis, need_tool=False),  # 使用 dict() 代替大括号 {}
                result=analysis,
                payload=choice
            ))
            return

        # ── 需要工具 ──
        self.logger.debug(f"[{self.name}] Thinking 模式：需要工具，执行工具调用")

        # 先 yield 思考过程（THOUGHT），让外层记录分析到短期记忆
        yield BaseState(StateCode.THOUGHT, ToolCallState(
            None, self._structured_output_fn,
            dict(analysis=analysis, need_tool=True),
            result=analysis,
            payload=choice
        ))

        # ── 执行本次返回的所有工具（使用公共方法） ──
        tool_states = self._execute_tool_states(choice.tool_calls)
        if not tool_states:
            self.logger.error(f"[{self.name}] Thinking 模式：模型分析了需要工具但实际未调用")
            yield BaseState(StateCode.Finish, analysis or "处理完成")
            return

        yield BaseState(StateCode.TOOL_CALL, tool_states)

    # ────────────────────────────── 非 Thinking 模式调用 ─────────────────────────

    def _wo_thinking_invoke(
            self,
            messages: List[Dict[str, Any]],
            tool_choice: Union[str, Dict],
            temperature: float,
            force_tool_choice: bool = True,
            **llm_client_kwargs,
    ) -> Iterator[BaseState]:
        """
        非 thinking 模式调用策略（原始两步调用）。

        Step 1 — 结构化分析
            强制/引导模型调用 final_structured_output，输出 analysis + need_tool。
        Step 2 — 工具执行
            need_tool=True  → 执行工具调用，yield TOOL_CALL
            need_tool=False → yield Finish（直接返回 analysis）
        """
        current_messages = list(messages)

        # ── Step 1: 结构化分析 ──
        analysis, need_tool, _tc_id, _last_choice = self._do_structured_analysis(
            messages, temperature, force_tool_choice, **llm_client_kwargs
        )
        self.logger.debug(f"[{self.name}] 分析结果: need_tool={need_tool}")

        thought_msg = Message(content=analysis, role="assistant")
        current_messages.append(thought_msg.to_openai_dict())
        structured_tc_state: ToolCallState = ToolCallState(
            _tc_id, self._structured_output_fn,
            {"analysis": analysis, "need_tool": need_tool},
            result=analysis,
            payload=_last_choice
        )

        if not need_tool:
            self.logger.debug(f"[{self.name}] 无需工具调用，直接返回结果")
            yield BaseState(StateCode.Finish, structured_tc_state)
            return

        self.logger.debug(f"[{self.name}] 需要工具调用，进入工具调用流程")
        yield BaseState(StateCode.THOUGHT, structured_tc_state)

        # ── Step 2: 工具执行 ──
        tool_states = self._execute_tool_calls(current_messages, tool_choice, temperature, **llm_client_kwargs)
        if not tool_states:
            self.logger.warning(
                f"[{self.name}] Step 1 分析需要工具，但 Step 2 LLM 实际未产生工具调用，直接返回"
            )
            yield BaseState(StateCode.Finish, analysis or "处理完成")
            return

        yield BaseState(StateCode.TOOL_CALL, tool_states)

    # ────────────────────────────── 核心流程（编排）─────────────────────────────

    def invoke_with_tools(
            self,
            messages: List[Dict[str, Any]],
            tool_choice: Union[str, Dict] = "auto",
            need_thought: bool = True,
            force_tool_choice: bool = True,
            **llm_client_kwargs
    ) -> Iterator[BaseState]:
        """
        核心调用流程（策略路由）。

        根据 self.supports_thinking + need_thought 自动选择策略：
        ─ supports_thinking=True 且 need_thought=True  → _thinking_invoke()  单次调用
        ─ supports_thinking=False 且 need_thought=True  → _wo_thinking_invoke() 两步调用
        ─ need_thought=False                              → 直接执行工具（跳过思考）

        :param messages:          OpenAI 格式的消息列表
        :param tool_choice:       工具选择策略（默认 "auto"）
        :param need_thought:      是否执行思考分析；为 False 时直接跳到工具调用步骤
        :param force_tool_choice: Step 1 是否强制指定 tool_choice。
                                  某些模型（如 qwen thinking mode）不支持，需设为 False。
                                  注意：thinking 模式下此参数不生效。
        :param kwargs:            额外 LLM 参数（temperature / max_tokens 等）
        :return:                  BaseState 包含 THOUGHT(思考过程) / TOOL_CALL(工具执行结果) / Finish(结束)
        """
        temperature = llm_client_kwargs.get("temperature", self.temperature)
        current_messages = list(messages)

        # ── Thinking 模式：单次调用 ──
        if self.supports_thinking and need_thought:
            yield from self._thinking_invoke(current_messages, tool_choice, temperature, **llm_client_kwargs)
            return

        # ── 非 thinking 模式：两步调用 ──
        if need_thought:
            yield from self._wo_thinking_invoke(
                current_messages, tool_choice, temperature, force_tool_choice, **llm_client_kwargs
            )
            return

        # ── 无思考步骤：直接执行工具 ──
        tool_states = self._execute_tool_calls(current_messages, tool_choice, temperature, **llm_client_kwargs)
        if not tool_states:
            yield BaseState(StateCode.Finish, "处理完成")
            return
        yield BaseState(StateCode.TOOL_CALL, tool_states)
