import json
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

COLLECTION_NAME = "math_problems"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text" 

INITIAL_DATA = [
    {
        "question": "Solve for x in the equation 2x + 5 = 15.",
        "solution": "1. Subtract 5 from both sides: 2x = 10.\n2. Divide both sides by 2: x = 5.\n3. The final answer is x=5. This is a simple linear algebra problem."
    },
    {
        "question": "What is the formula for calculating the area of a trapezoid?",
        "solution": "The area (A) of a trapezoid is calculated by adding the two parallel sides (a and b), dividing by two, and then multiplying by the height (h). Formula: A = (1/2) * (a + b) * h. This formula is derived from splitting the trapezoid into two triangles and a rectangle."
    },
    {
        "question": "Find the derivative of f(x) = x^3.",
        "solution": "1. Apply the power rule: d/dx(x^n) = n*x^(n-1).\n2. Set n=3: d/dx(x^3) = 3 * x^(3-1).\n3. The derivative is 3x^2. This is a fundamental concept in single-variable calculus."
    }
]

def create_kb():
    """Loads math data and creates a ChromaDB vector store."""
    if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
        print(f"✅ Knowledge Base already exists at {CHROMA_PATH}. Skipping setup.")
        return

    print("Loading data for KB indexing...")
    
    documents = [item['solution'] for item in INITIAL_DATA]
    metadatas = [{"question": item['question']} for item in INITIAL_DATA]
    
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ Knowledge Base created successfully at {CHROMA_PATH}")

if __name__ == "__main__":
    with open("math_data.json", 'w') as f:
        json.dump(INITIAL_DATA, f, indent=4)
        
    create_kb()
