
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from uuid import uuid4

load_dotenv()

quadrent_client = QdrantClient("http://localhost:6333")
if quadrent_client:
    print("Qdrant client is ready")

qdrant_vector_store = QdrantVectorStore(
    client=quadrent_client,
    collection_name="advanced-langchain",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

url = "http://localhost:6333"


if __name__ == "__main__":
    print("injesting....")
    loader = TextLoader("langchain_rag/medium1.txt")
    document = loader.load()

    print("splitting")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )

    chunkz = text_splitter.split_documents(document)

    embaddins = OpenAIEmbeddings(model="text-embedding-3-small")

    # ids = [str(uuid4()) for _ in range(len(chunkz))]
    # qdrant_vector_store.add_documents(documents=chunkz, ids=ids)
