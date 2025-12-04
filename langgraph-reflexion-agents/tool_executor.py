from answer_schema import AnswerQuestion, ReviseAnswer
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

load_dotenv()


tavily_tool = TavilySearch(max_result=5)


def run_queries(search_queries: list[str], **kwargs):
    """run the generate queries"""
    return tavily_tool.batch([{"query": query}] for query in search_queries)


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
