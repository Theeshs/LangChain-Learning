from __future__ import annotations

from typing import Any, Dict

from graph.state import retriver

from langgraph_corrective_rag.graph.state import GraphState


def retrive(state: GraphState) -> Dict[str, Any]:
    print("Retrive")
    question = state["question"]

    documents = retriver.invoke(question)
    return {"documents": documents, "question": question}
