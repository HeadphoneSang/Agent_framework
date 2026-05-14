import redis
import json
from typing import Sequence, List
from config import RedisConfig

_config = RedisConfig()

_pool = redis.ConnectionPool(
    host=_config.get("redis_host", "localhost"),
    port=_config.get("redis_port", 6379),
    db=0,
    max_connections=50,
    decode_responses=False
)


def _get_conn() -> redis.Redis:
    return redis.Redis(connection_pool=_pool)


_client: redis.Redis = _get_conn()


def set_value(key, value):
    _client.set(key, value)


def get_value(key):
    return _client.get(key)


def get_all_list(key: str) -> List[str]:
    return _client.lrange(key, 0, -1)


def push_to_list_right(key, items: Sequence[dict]) -> int:
    """
    向列表的右侧添加元素
    :param key: jian
    :param items: 添加的元素
    :return: 当前list的长度
    """
    json_list = [json.dumps(item) for item in items]
    return _client.rpush(key, *json_list)


def rm_key(key):
    _client.delete(key)


