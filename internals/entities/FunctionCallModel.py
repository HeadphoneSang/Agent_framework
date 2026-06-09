from pydantic import BaseModel, Field


class FunctionCallModel(BaseModel):
    analysis: str = Field(
        description="思考过程，比如对于当前问题的回答或者是解决问题的思路。该分析可以直接根据已有信息回答问题，也可以为下文的具体工具调用提供调用的信息。")
    need_tool: bool = Field(description="是否需要调用工具")