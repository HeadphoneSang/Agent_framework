from typing import Dict, Any


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def format_messages(input_params: Dict[str, Any], prompt_template: str, session_memories: list[dict[str, str]]):
    messages = [
        {
            'role': 'user',
            'content': prompt_template.format_map(SafeDict(**input_params))
        },
        {
            'role': 'system',
            'content': '## 下文是当前会话的短期聊天记录'
        },
        *session_memories
    ]
    return messages
