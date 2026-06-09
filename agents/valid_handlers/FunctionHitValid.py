import json

from openai.types.chat import ChatCompletionMessage

from agents.channels import BaseValidHandler
from internals.tool.StructuredFunctionTool import StructuredFunctionTool


class FunctionHitValid(BaseValidHandler):

    @staticmethod
    def _diagnose_onread(llm_message: ChatCompletionMessage, function_name: str) -> str:
        """
        诊断模型为什么没有调用目标函数，返回具体的错误描述。
        分三种情况：直接输出文本、调用了错误的函数、什么函数都没调。
        """
        # 情况1：model output is null or empty — 极罕见的边缘情况
        if not llm_message.tool_calls:
            content = (llm_message.content or "").strip()
            if content:
                preview = content[:300]
                return (
                    f"## 错误：你直接输出了文本，没有调用任何函数\n\n"
                    f"你输出的内容开头：\"{preview}\"\n\n"
                    f"### 正确做法\n"
                    f"不要输出文本，而是通过 tool_calls 调用 `{function_name}` 函数。输出的内容填写在参数里面 。"
                )
            else:
                return (
                    f"## 错误：你的回复中没有包含任何有效的函数调用\n\n"
                    f"### 正确做法\n"
                    f"通过 tool_calls 调用 `{function_name}` 函数，"
                    f"并传入 analysis（你的分析）和 need_tool（是否需要工具）两个参数。"
                )

        # 情况2：调用了函数，但不是目标函数
        called_names = [tc.function.name for tc in llm_message.tool_calls]
        return (
            f"## 错误：你试图调用 {called_names}，但没有调用 `{function_name}`\n\n"
            f"当前步骤**只允许**调用 `{function_name}` 一个函数，其他函数将在后续步骤中使用。\n\n"
            f"### 请重新调用\n"
            f"```\n"
            f"{function_name}(analysis=\"你的思考过程或最终答案\", need_tool=true/false)\n"
            f"```"
        )

    @staticmethod
    def _build_correction_guidance(function_name: str, escalation_level: int = 0) -> str:
        """根据重试次数生成不同力度的纠正指引"""
        base = (
            f"\n\n### 纠正指引\n"
            f"请重新生成回复。你必须通过 tool_calls 调用 `{function_name}` 函数，"
            f"不要在 content 中输出文本，也不要调用其他函数。"
        )
        if escalation_level >= 2:
            json_schema = StructuredFunctionTool().to_openai_schema()
            json_str = json.dumps(json_schema)
            base += (
                f"\n\n### 明确示例（第 {escalation_level + 1} 次纠正）\n"
                f"你的回复应该严格遵循以下格式：\n"
                f"```json\n{json_str}"
                f"```\n"
                f"**不要输出任何多余文本，只调用这一个函数。**"
            )
        return base

    def onRead(self, llm_message: ChatCompletionMessage, **kwargs):
        function_name = kwargs.get("function_name", "")
        if not function_name:
            raise ValueError("FunctionHitValid 缺少 function_name 参数")

        # 没有 tool_calls → 模型直接输出文本/空回复
        if not llm_message.tool_calls:
            error_detail = self._diagnose_onread(llm_message, function_name)
            raise ValueError(error_detail)

        # 检查是否调用了目标函数
        for tc in llm_message.tool_calls:
            if tc.function.name == function_name:
                return True

        # 调用了函数但没命中目标
        error_detail = self._diagnose_onread(llm_message, function_name)
        raise ValueError(error_detail)

    def onError(
            self,
            _llm_message: ChatCompletionMessage,
            error: Exception,
            function_name: str = None,
            escalation_level: int = 0,
            **_kwargs
    ):
        """
        模型没有调用指定函数时，输出具体诊断 + 纠正指引。
        escalation_level 标识这是第几次重试（0=第一次），越往后指引越详细。
        """
        # 消费接口要求的参数（按抽象基类签名）
        _ = _llm_message, _kwargs

        error_msg = str(error)
        guidance = self._build_correction_guidance(function_name, escalation_level)
        raise ValueError(error_msg + guidance)

    def onSuccess(self, llm_message: ChatCompletionMessage, **kwargs):
        pass
