
from config import BaseConfig


class SystemConfig(BaseConfig):
    """系统配置类"""
    def __init__(self):
        super().__init__(
            "system_config.yml"
        )
