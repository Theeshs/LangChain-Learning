from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

if __name__ == "__main__":
    print("Hello..! from Corrective Rag")
    print("Asking question that related to things in vector store")
    print(app.invoke(input={"question": "what is agent memory"}))
    print("Asking a question based on a context that is not available in vector store")
    print(app.invoke(input={"question": "How to make pizza"}))
