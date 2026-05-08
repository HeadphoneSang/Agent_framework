import os

from config import BaseConfig
from utils.fileUtils import get_abs_path


class ProviderConfig(BaseConfig):
    """
    供应商的配置文件
    """

    def __init__(self):
        super().__init__("provider.yml")

    def get_config_path(self):
        return get_abs_path(os.path.join("internals", self.config_name))
