from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()


class Usernames(BaseModel):
    """Schema for a username of a user that identified by the agent"""

    username: str = Field(
        description="username of the users who eiter a follower or following")


class AgentResponse(BaseModel):
    """Schema from the agent"""

    followers: List[str] = Field(
        default_factory=list, description="the list of followers"
    )
    followings: List[str] = Field(
        default_factory=list, description="the list of followings"
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
tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main(username: str):
    query = f"""Search instagram for {username} on instagram and scrape the all followers
      and followings and return their instagram username separately
      """
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content=query
            )
        }
    )
    print(result["structured_response"])


if __name__ == "__main__":
    main("@hellotheesh")
