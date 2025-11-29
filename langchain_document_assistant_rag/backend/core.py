import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = os.environ["INDEX_NAME"]


def run_llm(query: str):
    embaddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embaddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_document_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_document_chain
    )

    result = qa.invoke(input={"input": query})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }

    # print(new_result)

    return new_result


# if __name__ == "__main__":
#     res = run_llm(query="What is LangChain chain?")

#     print(res)
