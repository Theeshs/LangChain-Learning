from pprint import pprint

from dotenv import load_dotenv
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (
    GradeHallucination,
    hallusination_grader_chain,
)
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from injestion import retriver

load_dotenv()


def test_retrival_grader_yes() -> None:
    qustion = "agent memory"
    docs = retriver.invoke(qustion)
    doc_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": qustion, "document": doc_text}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_no() -> None:
    qustion = "agent memory"
    docs = retriver.invoke(qustion)
    doc_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "How to make pizza", "document": doc_text}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriver.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})

    pprint(generation)


def test_hallucination_grader_anwer_yes() -> None:
    question = "agent memory"
    docs = retriver.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination = hallusination_grader_chain.invoke(
        {"documents": docs, "generation": generation}
    )

    assert res.binary_score


def test_hallucination_grader_anwer_yes() -> None:
    question = "agent memory"
    docs = retriver.invoke(question)
    res: GradeHallucination = hallusination_grader_chain.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )

    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})

    assert res.datasource == "vectorstore"


def test_router_to_web_search() -> None:
    question = "how to make pizza"
    res: RouteQuery = question_router.invoke({"question": question})

    assert res.datasource == "web_search"
