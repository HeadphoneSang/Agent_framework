import os

from openai import BaseModel

from utils.configUtils import load_yml
from utils.fileUtils import get_abs_path

config_root_path = get_abs_path("configs")


class BaseConfig(BaseModel):
    """
    基础配置类
    """

    def __init__(self, config_name: str):
        super().__init__(config_name=config_name)
        self.__load_config_disk__()

    def get_config_path(self):
        """
        获取配置文件路径
        :return: 配置文件路径
        """
        return os.path.join(config_root_path, self.config_name)

    def __load_config_disk__(self):
        """
        从磁盘加载配置文件
        :return: None
        """
        config_path = self.get_config_path()
        self.config_dict = load_yml(config_path)

    def get(self, key, default=None):
        """
        获取配置项
        :param default:
        :param key: 配置项key
        :return: 配置项value
        """
        value = self.config_dict.get(key)
        return value or default
