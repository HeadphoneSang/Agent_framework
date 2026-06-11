import os
import re
from typing import Optional

from .BaseConfig import BaseConfig, config_root_path


class DotenvConfig(BaseConfig):
    """
    .env 文件配置读取器

    继承 BaseConfig 的 get(key, default) 接口，
    同时将 .env 中的变量注入 os.environ，使 os.environ.get() 也能直接访问。

    支持:
    - KEY=VALUE 标准键值对
    - # 行内注释（值中不含 #）
    - 空行与 # 注释行自动跳过
    - 引号包裹的值自动剥除
    """

    def __init__(self, config_name: str = ".env"):
        self._parsed: dict[str, str] = {}
        super().__init__(config_name)

    # ── 覆写父类加载逻辑 ──────────────────────────────────────

    def __load_config_disk__(self):
        config_path = self.get_config_path()
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f".env 文件不存在: {config_path}")

        parsed: dict[str, str] = {}
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行与纯注释行
                if not line or line.startswith("#"):
                    continue
                # 只取 # 之前的部分作为有效值（行内注释）
                if "#" in line:
                    # 检查 # 是否在引号内部（简单处理：不在引号内才截断）
                    cleaned = _strip_inline_comment(line)
                else:
                    cleaned = line

                match = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", cleaned)
                if match:
                    key = match.group(1)
                    value = match.group(2)
                    # 剥除首尾引号
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                    parsed[key] = value

        self.config_dict = parsed
        self._parsed = parsed

        # 注入到进程环境变量（不覆盖已有的）
        for k, v in parsed.items():
            if k not in os.environ:
                os.environ[k] = v

    # ── 便利属性：按数据源分组 ────────────────────────────────

    @property
    def qdrant(self) -> "QdrantEnv":
        return QdrantEnv(self._parsed)

    @property
    def neo4j(self) -> "Neo4jEnv":
        return Neo4jEnv(self._parsed)

    @property
    def embed(self) -> "EmbedEnv":
        return EmbedEnv(self._parsed)


# ── 内联注释处理 ──────────────────────────────────────────────────


def _strip_inline_comment(line: str) -> str:
    """去掉行内注释：# 不在引号内时才截断。"""
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i].rstrip()
    return line


# ── 分组的类型安全访问器 ─────────────────────────────────────────


class QdrantEnv:
    """Qdrant 向量数据库配置"""

    def __init__(self, data: dict[str, str]):
        self.__data = data

    @property
    def url(self) -> Optional[str]:
        return self.__data.get("QDRANT_URL")

    @property
    def api_key(self) -> Optional[str]:
        return self.__data.get("QDRANT_API_KEY")

    @property
    def collection(self) -> Optional[str]:
        return self.__data.get("QDRANT_COLLECTION")

    @property
    def vector_size(self) -> int:
        val = self.__data.get("QDRANT_VECTOR_SIZE", "1024")
        try:
            return int(val)
        except ValueError:
            return 1024

    @property
    def distance(self) -> str:
        return self.__data.get("QDRANT_DISTANCE", "cosine")

    @property
    def timeout(self) -> int:
        val = self.__data.get("QDRANT_TIMEOUT", "30")
        try:
            return int(val)
        except ValueError:
            return 30


class Neo4jEnv:
    """Neo4j 图数据库配置"""

    def __init__(self, data: dict[str, str]):
        self.__data = data

    @property
    def uri(self) -> Optional[str]:
        return self.__data.get("NEO4J_URI")

    @property
    def username(self) -> Optional[str]:
        return self.__data.get("NEO4J_USERNAME")

    @property
    def password(self) -> Optional[str]:
        return self.__data.get("NEO4J_PASSWORD")

    @property
    def database(self) -> Optional[str]:
        return self.__data.get("NEO4J_DATABASE")

    @property
    def max_connection_lifetime(self) -> int:
        val = self.__data.get("NEO4J_MAX_CONNECTION_LIFETIME", "3600")
        try:
            return int(val)
        except ValueError:
            return 3600

    @property
    def max_connection_pool_size(self) -> int:
        val = self.__data.get("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")
        try:
            return int(val)
        except ValueError:
            return 50

    @property
    def connection_timeout(self) -> int:
        val = self.__data.get("NEO4J_CONNECTION_TIMEOUT", "60")
        try:
            return int(val)
        except ValueError:
            return 60


class EmbedEnv:
    """嵌入模型配置"""

    def __init__(self, data: dict[str, str]):
        self.__data = data

    @property
    def model_type(self) -> Optional[str]:
        return self.__data.get("EMBED_MODEL_TYPE")

    @property
    def model_name(self) -> Optional[str]:
        return self.__data.get("EMBED_MODEL_NAME") or None

    @property
    def api_key(self) -> Optional[str]:
        return self.__data.get("EMBED_API_KEY")

    @property
    def base_url(self) -> Optional[str]:
        return self.__data.get("EMBED_BASE_URL") or None


# ── 独立函数：快速加载到 os.environ ─────────────────────────────


def load_dotenv(env_path: Optional[str] = None, override: bool = False) -> dict[str, str]:
    """
    轻量加载 .env 到 os.environ，不依赖 DotenvConfig 类。

    Args:
        env_path: .env 文件路径，默认 configs/.env
        override: 是否覆盖已存在的环境变量

    Returns:
        解析出的键值对字典
    """
    if env_path is None:
        env_path = os.path.join(config_root_path, ".env")

    if not os.path.isfile(env_path):
        raise FileNotFoundError(f".env 文件不存在: {env_path}")

    parsed: dict[str, str] = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = _strip_inline_comment(line)
            match = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", line)
            if match:
                key = match.group(1)
                value = match.group(2)
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                parsed[key] = value

    for k, v in parsed.items():
        if override or k not in os.environ:
            os.environ[k] = v

    return parsed


# ── 快捷使用 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    env = DotenvConfig()
    print("=== Qdrant ===")
    print(f"  URL:        {env.qdrant.url}")
    print(f"  Collection: {env.qdrant.collection}")
    print(f"  VectorSize: {env.qdrant.vector_size}")
    print(f"  Distance:   {env.qdrant.distance}")
    print()
    print("=== Neo4j ===")
    print(f"  URI:      {env.neo4j.uri}")
    print(f"  User:     {env.neo4j.username}")
    print(f"  Database: {env.neo4j.database}")
    print()
    print("=== Embed ===")
    print(f"  Type:  {env.embed.model_type}")
    print(f"  Name:  {env.embed.model_name}")
    print(f"  HasKey: {bool(env.embed.api_key)}")
    print()
    print("=== 通用 get(key) ===")
    print(f"  QDRANT_URL  = {env.get('QDRANT_URL')}")
    print(f"  NEO4J_URI   = {env.get('NEO4J_URI')}")
    print(f"  EMBED_API_KEY = {env.get('EMBED_API_KEY')}")
