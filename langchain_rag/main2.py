from dotenv import load_dotenv
from injestion import qdrant_vector_store
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retreaving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    retreaval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer

    {context}

    Question: {question}
    Helpful Answer:
    """

    custom_rag_promt = PromptTemplate.from_template(template)
    # combin_docs_chain = create_stuff_documents_chain(
    #     llm, retreaval_qa_chat_prompt)
    # retraval_chain = create_retrieval_chain(
    #     retriever=qdrant_vector_store.as_retriever(), combine_docs_chain=combin_docs_chain
    # )
    rag_chain = (
        {
            "context": qdrant_vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_promt
        | llm
    )

    result = rag_chain.invoke("What is Pinecone?")

    print(result)
