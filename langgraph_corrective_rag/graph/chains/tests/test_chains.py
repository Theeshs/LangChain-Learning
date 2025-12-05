from pprint import pprint

from dotenv import load_dotenv
from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from injestion import retriver

load_dotenv()


def test_retrival_grader_yes() -> None:
    qustion = "agent memory"
    docs = retriver.invoke(qustion)
    print(docs)
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
