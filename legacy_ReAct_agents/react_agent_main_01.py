from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents import AgentExecutor
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from langchain_classic import hub

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    print("Runnint react agent")
    result = chain.invoke(
        {
            "input": {
                "input": "search for 3 job postings for an ai engineer using langchain in Sweden on linkedin and list their details?"
            }
        }
    )
    print(result)


if __name__ == "__main__":
    main()
