from config import BaseConfig


class LogConfig(BaseConfig):
    """
    日志配置类
    """
    def __init__(self):
        super().__init__("logger_config.yml")