import traceback

from openai.types.chat import ChatCompletionMessage
from agents.channels import BaseValidHandler
from logger.loggerUtil import get_logger


class ValidPipline:
    """
    A pipeline of validators
    """
    handlers: list[BaseValidHandler]
    logger: any

    def __init__(self, handlers: list = [], logger: any = None):
        self.handlers = handlers
        self.logger = logger or get_logger()

    def append(self, handler: BaseValidHandler):
        self.handlers.append(handler)

    def validate(self, llm_message: ChatCompletionMessage, **kwargs):
        next0 = True
        for handler in self.handlers:
            try:
                next0: bool = handler.onRead(llm_message, **kwargs)
                handler.onSuccess(llm_message, **kwargs)
            except Exception as e:
                next0: bool = handler.onError(llm_message, e, **kwargs)
                self.logger.error(f"模型返回结果错误: \n{traceback.format_exc()}")
            finally:
                if not next0:
                    break
