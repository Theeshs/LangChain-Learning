from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    Represents the state of the graph

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list_of_documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
