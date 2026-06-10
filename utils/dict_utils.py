def union_dict(dict1, dict2):
    new_dict = dict1.copy()  # 先浅拷贝一份
    new_dict.update(dict2)
    return new_dict
