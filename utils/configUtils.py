import os.path

from .fileUtils import get_abs_path


def load_yml(file_path):
    """
    加载yml文件
    :param file_path:
    :return:
    """
    import yaml
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_provider_yml():
    provider_path = get_abs_path(os.path.join('internals', 'provider.yml'))
    return load_yml(provider_path)


provider_yml: dict = load_provider_yml()
