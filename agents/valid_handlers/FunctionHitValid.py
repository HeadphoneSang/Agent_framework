from openai.types.chat import ChatCompletionMessage
from pydantic import ValidationError

from agents.channels import BaseValidHandler


class FunctionHitValid(BaseValidHandler):

    @staticmethod
    def _check_has_function(llm_message: ChatCompletionMessage, function_name: str, **kwargs):
        """
        检查模型是否调用了结构化输出的函数，同时结构化输出的函数是否输出正确
        :param llm_message:
        :param function_name:
        :param kwargs:
        :return:
        """
        tc_list = llm_message.tool_calls
        for tc in tc_list:
            tc_name = tc.function.name
            if tc_name == function_name:
                return True
        return False

    def onRead(self, llm_message: ChatCompletionMessage, **kwargs):
        if self._check_has_function(llm_message, **kwargs):
            return True
        else:
            raise ValueError("你的回复格式错误！你必须调用结构化输出函数，来输出你的思路和是否需要调用工具！请重新回答。")

    def onError(self, llm_message: ChatCompletionMessage,error: Exception,function_name: str = None,  **kwargs):
        """
        没有调用结构化输出函数，模型出现幻觉，抛出错误
        :param function_name: 结构化函数名称
        :param llm_message:
        :param error:
        :param kwargs:
        :return:
        """
        error_msg = str(error)
        error_msg = f"{error_msg} \n 你应该调用结构化输出函数：{function_name} 来回答问题！"
        raise ValueError(error_msg)

    def onSuccess(self, llm_message: ChatCompletionMessage, **kwargs):
        pass
