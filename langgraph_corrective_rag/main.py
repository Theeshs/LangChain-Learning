from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

if __name__ == "__main__":
    print("Hello..! from Corrective Rag")
    # print(app.invoke(input={"question": "what is agent memory"}))
    print(app.invoke(input={"question": "How to make pizza"}))
