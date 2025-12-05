from dotenv import load_dotenv
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


llm = ChatOpenAI(temperature=0)
propmpt = hub.pull("rlm/rag-prompt")

generation_chain = propmpt | llm | StrOutputParser()
