from __future__ import annotations

from typing import Any, Dict

from dotenv import load_dotenv
from langchain_classic.schema import Document
from langchain_tavily import TavilySearch

from langgraph_corrective_rag.graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("--- Web Search ---")
    question = state["question"]
    documents = state["documents"]

    tavily_search_results = web_search_tool.invoke({"query": question})
    joined_taviliy_result = "\n".join(
        [
            tavily_search_result["content"]
            for tavily_search_result in tavily_search_results
        ]
    )
    web_result = Document(page_content=joined_taviliy_result)
    if documents is not None:
        documents.append(web_result)
    else:
        documents = [web_result]

    return {"documents": documents, "question": question}
