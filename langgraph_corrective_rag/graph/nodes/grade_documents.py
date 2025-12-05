from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("-- Check Document relevenc question")
    question = state["question"]
    document = state["documents"]

    filterd_docs = []
    web_search = False

    for d in document:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("--- Grade Document Relevent ---")
            filterd_docs.append(d)
        else:
            print("--- Grade Document Not Relevent ---")
            web_search = True
            continue

    return {"documents": filterd_docs, "question": question, "web_search": web_search}
