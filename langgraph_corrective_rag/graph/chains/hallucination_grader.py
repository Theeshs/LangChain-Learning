from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(temperature=0)


class GradeHallucination(BaseModel):
    """Binary score for hallucination present in generation answer"""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes', 'no'",
    )


structrued_llm_grader = llm.with_structured_output(GradeHallucination)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallusination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallusination_grader_chain: RunnableSequence = (
    hallusination_prompt | structrued_llm_grader
)
