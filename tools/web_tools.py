import os
from typing import List, Dict, Any

from tavily import TavilyClient

from internals.tool import Tool
from internals.tool.Tool import ToolParameter


class SearchTool(Tool):
    """时间工具"""

    def get_params(self) -> List[ToolParameter]:
        pass

    def __init__(self):
        super().__init__(name="search",
                         description="从网络中获取相关内容，参数：query[str]，返回：字符串内容")

    def run(self, params: Dict[str, Any]) -> str:
        return search(**params)


def search(query) -> str:
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"
    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)
    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]
        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        if not formatted_results:
            return "抱歉，没有找到相关的信息"
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"
