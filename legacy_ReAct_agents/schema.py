from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for source used by the agent"""

    ur: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """Shcema for agent response with the answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )
