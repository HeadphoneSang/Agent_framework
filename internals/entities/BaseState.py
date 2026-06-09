from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class StateCode(IntEnum):
    INIT = 0
    SUCCESS = 1
    Finish = 2
    FAILED = 3
    TOOL_CALL = 4
    THOUGHT = 5


@dataclass
class BaseState:
    """
    基础状态类
    """
    state: int
    payload: Any

    def __eq__(self, other):
        if not isinstance(other, BaseState) and not isinstance(other, StateCode):
            return False
        if isinstance(other, StateCode):
            return self.state == other.value
        return self.state == other.state

