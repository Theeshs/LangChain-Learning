from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()


class Source(BaseModel):
    """Schema for a souce used by the agent"""

    url: str = Field(description="url of the soruce")


class AgentResponse(BaseModel):
    """Schema from the agent"""

    answer: str = Field("The anget's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="the list of sources to generate the answer"
    )


tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that sarchs over the internet

    Args:
        query: The query to search for

    Returns:
        The search result
    """
    print(f"Searching intenet for {query}")
    return tavily.search(query=query)


llm = ChatOpenAI()
tools = [search]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in Sweden on linkedin and list their details?"
            )
        }
    )
    print(result["structured_response"])


if __name__ == "__main__":
    main()
