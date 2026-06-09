import os
import traceback
from typing import List, Dict

from openai import OpenAI
from openai.types.chat import ChatCompletion

from logger.loggerUtil import get_logger


class HelloAgentsLLM:
    """
    A simple LLM that uses the OpenAI API to generate responses.
    """

    def __init__(self, api_key: str, base_url: str, timeout: int = 60, model: str = "deepseek-chat", print_content:bool=False):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.logger = get_logger()
        self.print_content = print_content

    def think(self, message: List[Dict[str, str]], temperature: float = 0.7, stream: bool = False, **client_kwargs) -> str:
        """
        Generate a response to the given message.
        """
        self.logger.debug(f"正在调用 {self.model} 模型...")

        if stream:
            response_content = self.stream_think(message, temperature, **client_kwargs)
        else:
            response_content = self.normal_think(message, temperature, **client_kwargs)
        self.logger.debug(f"{self.model} 模型返回: {response_content}")
        return response_content

    def normal_think0(self, message: List[Dict[str, str]], **client_kwargs) -> ChatCompletion:
        """
        Generate a response to the given message.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                stream=False,
                **client_kwargs
            )
            return response
        except Exception:
            self.logger.error(f"模型调用失败\n{traceback.format_exc()}")
            raise

    def think_origin(self, message: List[Dict[str, str]], **client_kwargs):
        """
        Generate a response to the given message.
        """
        self.logger.debug(f"正在调用 {self.model} 模型...")
        response_content = self.normal_think0(message, **client_kwargs)
        self.logger.debug(f"{self.model} 模型返回: {response_content}")
        return response_content

    def normal_think(self, message: List[Dict[str, str]], temperature: float = 0.7, **client_kwargs) -> str:
        """
        Generate a response to the given message.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                stream=False,
                **client_kwargs
            )
            content = response.choices[0].message.content
            if self.print_content:
                print(content)
            return content
        except Exception:
            self.logger.error(f"模型调用失败\n{traceback.format_exc()}")
            raise

    def stream_think(self, message: List[Dict[str, str]], temperature: float = 0.7, **client_kwargs):
        """
                Generate a response to the given message.
                """

        def storage_chunk_stream(stream_response, chunk_list0):
            for chunk0 in stream_response:
                if not chunk0.choices:
                    continue  # 跳过 usage 等无内容的信号 chunk
                content0 = chunk0.choices[0].delta.content or ""
                chunk_list0.append(content0)
                yield content0

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                stream=True,
                **client_kwargs
            )
            chunk_list = []
            for chunk_content in storage_chunk_stream(response, chunk_list):
                if self.print_content:
                    print(chunk_content, end="", flush=True)
            if self.print_content:
                print("\n")
            return "".join(chunk_list)
        except Exception:
            self.logger.error(f"模型调用失败\n{traceback.format_exc()}")
            raise

