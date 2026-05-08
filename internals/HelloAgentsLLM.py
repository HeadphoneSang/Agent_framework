import os
from typing import List, Dict

from openai import OpenAI
from logger.loggerUtil import get_logger


class HelloAgentsLLM:
    """
    A simple LLM that uses the OpenAI API to generate responses.
    """

    def __init__(self, api_key: str, base_url: str, timeout: int = 60, model: str = "deepseek-chat", ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.logger = get_logger()

    def think(self, message: List[Dict[str, str]], temperature: float = 0.7, stream: bool = False) -> str:
        """
        Generate a response to the given message.
        """
        self.logger.debug(f"正在调用 {self.model} 模型...")

        if stream:
            response_content = self.stream_think(message, temperature)
        else:
            response_content = self.normal_think(message, temperature)
        self.logger.debug(f"{self.model} 模型返回: {response_content}")
        return response_content

    def normal_think(self, message: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Generate a response to the given message.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                stream=False,
            )
            content = response.choices[0].message.content
            print(content)
            return content
        except Exception as e:
            self.logger.error(f"模型调用失败: {e}")
            return "模型调用失败"

    def stream_think(self, message: List[Dict[str, str]], temperature: float = 0.7):
        """
                Generate a response to the given message.
                """

        def storage_chunk_stream(stream_response, chunk_list0):
            for chunk0 in stream_response:
                content0 = chunk0.choices[0].delta.content or ""
                chunk_list0.append(content0)
                yield content0

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                stream=True,
            )
            chunk_list = []
            for chunk_content in storage_chunk_stream(response, chunk_list):
                print(chunk_content, end="", flush=True)
            print("\n")
            return "".join(chunk_list)
        except Exception as e:
            self.logger.error(f"模型调用失败: {e}")
            return "模型调用失败"

if __name__ == "__main__":
    api_key = os.environ['DEEPSEEK_API_KEY']
    base_url = os.environ['DEEPSEEK_BASE_URL']
    agent_client = HelloAgentsLLM(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": "你是一个助手，请根据用户输入的指令进行回答。",
        },
        {
            "role": "user",
            "content": "请帮我介绍一下徐州",
        }
    ]
    get_logger().info(f"agent 的完整回答如下: {agent_client.think(messages, temperature=0.7)}")
