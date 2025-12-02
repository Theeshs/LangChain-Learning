from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools

load_dotenv()

SYSTETM_MESSAGE = """
You are a helpful assistant that can use tools to answer the questions
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
        Run the agent reasoning node
    """

    response = llm.invoke(
        {"role": "system", "content": SYSTETM_MESSAGE}, *state["messages"])
    return {"messages", [response]}


tool_node = ToolNode(tools)
