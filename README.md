# MathProf


Agentic RAG system for intelligent math problem-solving with multi-source retrieval, powered by local LLMs via Ollama.

## Features

- ✅ **Local LLM Support** - Works with any Ollama-compatible model
- ✅ **Agentic Workflow** - LangGraph orchestration with intelligent routing
- ✅ **Multi-Source Retrieval** - Vector database + Web search fallback
- ✅ **Self-Correction** - DSPy-based confidence scoring
- ✅ **LaTeX Rendering** - Beautiful mathematical notation display
- ✅ **Human-in-the-Loop** - Flags low-confidence responses for review

## Architecture

```
User Query → Input Guardrail → Router Agent → [KB Retrieval | Web Search] 
          → Generation Agent → Output Guardrail → Response
```

**Tech Stack:**
- **Backend**: FastAPI, LangChain, LangGraph, DSPy
- **LLM Runtime**: Ollama (local inference)
- **Vector DB**: ChromaDB with Nomic embeddings
- **Web Search**: DuckDuckGo + BeautifulSoup
- **Frontend**: React + KaTeX

## Prerequisites

### 1. Install Ollama
Download from [ollama.com](https://ollama.com)

### 2. Pull a Model
Choose any model you prefer:

# Recommended: IBM Granite (3B parameters)
ollama pull granite3.1:3b

# Alternatives:
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull qwen2.5:3b
```

### 3. Pull Embedding Model

ollama pull nomic-embed-text
```

### 4. Install Python Dependencies
```
cd backend
pip install -r requirements.txt
```
### 5. Install Node.js
Requires Node.js 16+ for the React frontend.

## Setup

### Step 1: Configure the Model
Edit `backend/agent_logic.py`:
```python
LLM_MODEL = "granite3.1:3b"  # Change to your preferred model
KB_EMBED_MODEL = "nomic-embed-text"
```

### Step 2: Initialize Knowledge Base
```bash
cd backend
python kb_setup.py
```
This creates a ChromaDB vector store with sample math problems.

### Step 3: Start Backend
```bash
python main.py
```
Backend runs on `http://localhost:8000`

### Step 4: Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on `http://localhost:5173`

## Usage

1. Open `http://localhost:5173` in your browser
2. Enter a math question (e.g., "Solve 3x + 9 = 21")
3. The agent will:
   - Route to knowledge base for textbook problems
   - Use web search for complex/recent topics
   - Generate step-by-step solutions with LaTeX formatting
   - Flag low-confidence answers for review

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI server
│   ├── agent_logic.py       # LangGraph workflow + DSPy
│   ├── kb_setup.py          # Vector DB initialization
│   ├── chroma_db/           # Persistent vector store
├── frontend/
│   ├── src/
│   │   └── App.jsx          # React UI with KaTeX
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Configuration

### Change LLM Model
In `agent_logic.py`:
```python
LLM_MODEL = "your-model-name"  # Must be pulled via Ollama
```

### Adjust Confidence Threshold
In `agent_logic.py`:
```python
def output_guardrail(state: AgentState):
    if state['confidence'] < 0.75:  # Change threshold here
        return {"route": "HUMAN_IN_LOOP"}
```

### Add Custom Knowledge
Add documents to `backend/math_data.json` and run:
```bash
python kb_setup.py
```

## How It Works

1. **Input Guardrail**: Validates math-related queries or general day-to-day queries
2. **Router Agent**: Decides between KB retrieval or web search
3. **Retrieval**: Fetches context from ChromaDB or DuckDuckGo
4. **Generation**: LLM creates step-by-step solution
5. **Output Guardrail**: Confidence scoring triggers self-correction if needed
6. **Frontend**: Renders LaTeX math notation via KaTeX


**Hardware:**
- 8GB+ RAM (16GB recommended)
- Optional: GPU for faster inference (4GB Recommended)

**Software:**
- Python 3.9+
- Node.js 16+
- Ollama

## License

MIT

## Credits

Built with:
- [Ollama](https://ollama.com) - Local LLM runtime
- [LangChain](https://langchain.com) - LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration
- [DSPy](https://github.com/stanfordnlp/dspy) - Self-correction
- [ChromaDB](https://www.trychroma.com/) - Vector database
