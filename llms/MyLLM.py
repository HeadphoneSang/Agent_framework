import os
from typing import Optional

from internals import HelloAgentsLLM
from logger.loggerUtil import get_logger
from config import ProviderConfig


class MyLLM(HelloAgentsLLM):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[str] = "auto",
            **kwargs
    ):
        # 处理无参数的自动匹配供应商
        api_key0 = None
        provider = provider.lower()
        self.provider =  provider
        self.config = ProviderConfig()
        try:
            self.logger = get_logger()
            if provider == "auto":
                # 便利所有的供应商，采用第一个可以用的
                provider_infos: dict = self.config.get("provider_list")
                for provider_name, provider_info in provider_infos.items():
                    api_key0 = os.environ.get(provider_info.get("key_name"))
                    if api_key0:
                        provider = provider_name
                        break
                model0 = provider_infos.get(provider).get("model_name")
                base_url0 = provider_infos.get(provider).get("base_url")
                self.logger.info(f"自动使用供应商: {provider}")
                self.logger.info(f"自动匹配使用模型: {model0}")
            else:
                model0 = self.config.get("provider_list").get(provider).get("model_name")
                base_url0 = self.config.get("provider_list").get(provider).get("base_url")
                api_key0 = os.environ.get(self.config.get("provider_list").get(provider).get("key_name"))
        except Exception as e:
            self.logger.error(f"获取供应商信息失败: {e}")
            model0 = None
            api_key0 = None
            base_url0 = None
        model = model or model0
        api_key = api_key or api_key0
        base_url = base_url or base_url0
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)


if __name__ == "__main__":
    llm = MyLLM()
    llm.think([{"role": "user", "content": "你叫什么名字？"}])
