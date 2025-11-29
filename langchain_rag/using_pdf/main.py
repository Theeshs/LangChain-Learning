import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Hi")
    loader = PyPDFLoader("langchain_rag/using_pdf/react.pdf")
    content = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)

    docs = text_splitter.split_documents(content)
    print(docs)
    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retreaval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combin_docs_chain = create_stuff_documents_chain(OpenAI(), retreaval_qa_chat_prompt)

    retreival_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combin_docs_chain
    )

    res = retreival_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
