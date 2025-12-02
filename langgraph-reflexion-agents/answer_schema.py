from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Anser the question"""

    answer: str = Field("~250 world detaild answer to the question")
    reflection: Reflection = Field(description=" Your reflection on the initial answer")
    search_queries: List[str] = Field(
        description="1-3 Search question for reasearching improvments to address teh critique of your current answer"
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""

    references: List[str] = Field(
        description="Citations motivations your updated answer"
    )
