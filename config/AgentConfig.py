from config import BaseConfig


class AgentConfig(BaseConfig):
    """
    智能体配置类
    """
    def __init__(self):
        super().__init__("agent_config.yml")
