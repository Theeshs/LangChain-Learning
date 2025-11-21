from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents import AgentExecutor
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()


def main():
    print("Runnint react agent")


if __name__ == "__main__":
    main()
