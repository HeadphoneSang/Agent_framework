from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from pydantic import ValidationError
from internals.entities import FunctionCallModel
from agents.channels import BaseValidHandler


class StructValidHandler(BaseValidHandler):

    def onRead(self, llm_message: ChatCompletionMessage, json_content: str = None, **kwargs):
        FunctionCallModel.model_validate_json(json_content)

    def onError(self, llm_message: ChatCompletionMessage, validError: ValidationError, function_name: str = None, **kwargs):
        error_header = f"{function_name} 函数的输入参数格式错误，具体的错误如下："
        error_items = []
        for error in validError.errors():
            location = " -> ".join(str(x) for x in error["loc"])
            reason = error["msg"]

            # 汉化一些常见的英文错误，让国产大模型理解更准确
            if error["type"] == "missing":
                reason = "这个字段是必填的，你漏掉了"
            elif error["type"] == "dict_type":
                reason = "格式错误，这里必须是一个 {} 字典键值对"
            elif error["type"] == "string_type":
                reason = "格式错误，这里必须是一个字符串"

            bad_input = error.get("input")
            error_items.append(f"- 错误位置 `[{location}]`: {reason}。你错误地输入了: `{bad_input}`")
        error_end_item = f"请检查你的输入参数，并重新调用函数：{function_name}。"
        error_msg = error_header + "\n".join(error_items) + "\n" + error_end_item
        raise ValueError(error_msg)

    def onSuccess(self, llm_message: ChatCompletionMessage, **kwargs):
        pass
