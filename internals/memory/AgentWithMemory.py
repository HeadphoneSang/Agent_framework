from typing import Dict, List, Any, Iterator
from agents.base import BaseAgent
from internals import Message
from internals.memory import BaseMemory

session_memories: Dict[str, BaseMemory] = {}


class AgentWithMemoryProxy(BaseAgent):
    """
    Agent with memory
    """

    def __init__(self, agent: BaseAgent, init_memory_func: callable):
        super().__init__(agent.name, agent.llm, agent.config)
        self._init_memory_func = init_memory_func
        self.agent = agent

    def stream(self, input_params: Dict[str, Any], question_key: str = "question", memories_key: str = "history",
               session_config=None, **kwargs) -> Iterator[Message]:
        if session_config is None:
            session_config = {"session_id": "default"}
        session_id = session_config.get("session_id", "default")
        memory = session_memories.get(session_id, None)
        if memory is None:
            memory = self._init_memory_func(session_config)
            session_memories[session_id] = memory
        persist_memories: List[Message] = memory.get_memories()
        history_prompt = "- " + "\n\n- ".join([f"[{msg.role}]: {msg.content}" for msg in persist_memories])
        input_params[memories_key] = history_prompt
        question = input_params.get(question_key, "")
        msg_iterator = self.agent.stream(input_params, **kwargs)
        # 初始化会话消息列表，用来最后的持久化
        session_msg_list: list[Message] = [Message(content=question, role="user")]
        for msg in msg_iterator:
            yield msg
            session_msg_list.append(msg)
        # 持久化所有会话记忆
        memory.add_memories(session_msg_list)

    def invoke(self, input_params: Dict[str, Any], question_key: str = "question", memories_key: str = "history",
               session_config=None, **kwargs) -> \
            list[Message]:
        if session_config is None:
            session_config = {"session_id": "default"}
        session_id = session_config.get("session_id", "default")
        memory = session_memories.get(session_id, None)
        if memory is None:
            memory = self._init_memory_func(session_config)
            session_memories[session_id] = memory
        persist_memories: List[Message] = memory.get_memories()
        # 将持久化内存添加到会话历史中
        history_prompt = "- " + "\n\n- ".join([f"[{msg.role}]: {msg.content}" for msg in persist_memories])
        input_params[memories_key] = history_prompt
        msg_list: list[Message] = self.agent.invoke(input_params=input_params, **kwargs)
        question: str = input_params.get(question_key, "")
        memory.add_memories([Message(content=question, role="user"), msg_list[-1]])
        return msg_list
