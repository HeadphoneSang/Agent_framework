from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletionMessage


class BaseValidHandler(ABC):
    @abstractmethod
    def onRead(self, llm_message: ChatCompletionMessage, **kwargs):
        """
        处理LLM返回的choice，对choice进行检查或者是预处理的处理器
        :param llm_message: LLM返回的choice
        :return: 返回true or false 决定是否继续执行
        """
        raise NotImplementedError

    @abstractmethod
    def onError(self, llm_message: ChatCompletionMessage, error: Exception, **kwargs):
        """
        这里一定要处理error，返回true or false决定是否继续执行
        :param llm_message: LLM返回的choice
        :param error: 错误
        :return: 无
        """
        raise NotImplementedError

    @abstractmethod
    def onSuccess(self, llm_message: ChatCompletionMessage, **kwargs):
        """
        当符合预期的时候，调用此函数，在onRead后执行
        :param llm_message: llm_message: LLM返回的choice
        :return: 无返回值
        """
        raise NotImplementedError
