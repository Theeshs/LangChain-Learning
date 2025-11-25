from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from injestion import qdrant_vector_store

from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    print("Retreaving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    retreaval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combin_docs_chain = create_stuff_documents_chain(
        llm, retreaval_qa_chat_prompt)
    retraval_chain = create_retrieval_chain(
        retriever=qdrant_vector_store.as_retriever(), combine_docs_chain=combin_docs_chain
    )

    result = retraval_chain.invoke(
        input={
            "input": "What is Pinecone?"
        }
    )

    print(result)
