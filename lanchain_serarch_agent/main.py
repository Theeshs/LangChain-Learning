from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#     Tool that sarchs over the internet

#     Args:
#         query: The query to search for

#     Returns:
#         The search result
#     """
#     print(f"Searching intenet for {query}")
#     return tavily.search(query=query)


llm = ChatOpenAI()
tools = [TavilySearch()]

agent = create_agent(
    model=llm, tools=tools
)


def main():
    result = agent.invoke({
        "messages": HumanMessage(
            content="search for 3 job postings for an ai engineer using langchain in Sweden on linkedin and list their details?"
        )
    })
    print(result)


if __name__ == "__main__":
    main()
