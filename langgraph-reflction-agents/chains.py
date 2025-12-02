from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()


reflection_propmpt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral intergram influencer grading a caption. Generate critique and recommendations for the user's caption."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generation_propmpt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a intagram photography influencer assistant tasked with writing excellent instagram post captions."
            " Generate the best instagram caption possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
generation_chain = generation_propmpt | llm
reflection_chain = reflection_propmpt | llm
