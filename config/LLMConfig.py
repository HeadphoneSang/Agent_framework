from config.BaseConfig import BaseConfig


class LLMConfig(BaseConfig):
    """
    LLM 配置类
    """

    def __init__(self):
        super().__init__("llm_config.yml")
