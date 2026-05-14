from config import BaseConfig


class RedisConfig(BaseConfig):
    """
    智能体配置类
    """
    def __init__(self):
        super().__init__("redis_config.yml")
