import os


def get_abs_path(path):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_path, path)


def load_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return ""
