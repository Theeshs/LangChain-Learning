from dotenv import load_dotenv
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import (
    hallusination_grader_chain as hallucination_grader,
)
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENT, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from langgraph.graph import END, StateGraph

load_dotenv()


def decide_to_generate(state):
    print("--- ASSESS GRADED DOCUMENt ---")
    if state["web_search"]:
        print("--- DECISION: NOT ALL DOCUMENTS RELEVEN TO QUESTION---")
        return WEBSEARCH
    else:
        print("--- DECISION: GENERATE ---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("--- Decision: Generation is grounded in documents ---")
        print("--- Grade Generation vs Question")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("--- Route Question ---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("--- ROUTE Question to Web Search ---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("--- ROUTE Question to RAG ---")
        return RETRIEVE


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENT, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question, {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE}
)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENT)
workflow.add_conditional_edges(
    GRADE_DOCUMENT, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"not supported": GENERATE, "useful": END, "not useful": WEBSEARCH},
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
