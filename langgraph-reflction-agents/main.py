import os
from typing import Annotated, TypedDict

from chains import generation_chain, reflection_chain
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    # return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_countinue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(
    GENERATE, should_countinue, path_map={END: END, REFLECT: REFLECT}
)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

if __name__ == "__main__":
    print("Hello Langgraph")
    # graph.get_graph().draw_mermaid()
    content = """
        During my travels in Huế 🇻🇳, I captured some beautiful candid moments — a couple walking in the hallway and a lady on a bench, lost in a call 😍. Wait till you see the final results!

        If you love travel and street photography, your support means the world 🙏 — a like, share, or comment helps me keep going. Let’s grow this journey together ❤️

        #photography #reels #instagood #travel #streetphotography #travelphotography #reach
        """

    inputs = {"messages": [HumanMessage(content=content)]}

    res = graph.invoke(inputs)

    print(res)
