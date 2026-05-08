import logging
import os
from datetime import datetime
from utils.fileUtils import get_abs_path
from config import LogConfig


config = LogConfig()
log_storage_path = config.get("log_storage_path")
LOG_ROOT_DIR = get_abs_path(log_storage_path)
LOG_LEVEL = logging.getLevelName(config.get("log_level"))

os.makedirs(LOG_ROOT_DIR, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(
        name: str = "agent",
        log_level: int = LOG_LEVEL,
        file_level: int = logging.DEBUG,
        log_file: str = None
) -> logging.Logger:
    """
    快速获得logger对象
    :param name: logger_id
    :param log_level: 控制台logger输出粒度
    :param file_level: 文件logger输出粒度
    :param log_file: log文件名称
    :return: 返回logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    if log_file is None:
        log_file = os.path.join(LOG_ROOT_DIR, f"{name}_{datetime.now().strftime('%Y%m%d%H%M')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)
    return logger
