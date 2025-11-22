from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schema import AgentResponse

load_dotenv()

# Tools
tools = [TavilySearch()]

# LLMs
llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(AgentResponse)

# Prompt
react_prompt = hub.pull("hwchase17/react")
# format_instructions = structured_llm.get_format_instructions()
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "tools", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")

# Agent
agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Chain
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm


def main():
    print("Running react agent")
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
