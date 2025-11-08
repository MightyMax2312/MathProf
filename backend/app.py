import os
import json
import requests
from typing import TypedDict, List, Dict, Any, Optional

# Limit GPU memory usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OLLAMA_NUM_GPU'] = '1'

# --- LLM & RAG COMPONENTS ---
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain_chroma import Chroma                         
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

# --- AGENT ORCHESTRATION & TOOLS ---
from langgraph.graph import StateGraph, END, START 
from ddgs import DDGS  
from bs4 import BeautifulSoup       

# --- SELF-CORRECTION (DSPy Bonus) ---
import dspy                         

# --- FASTAPI ---
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CONFIGURATION ---
LLM_MODEL = "granite4:3b"
KB_EMBED_MODEL = "nomic-embed-text" 
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "math_problems"

# Initialize Core Components (Lazy Initialization)
LLM = ChatOllama(model=LLM_MODEL, temperature=0.0)
MATH_RETRIEVER = None
DSPY_GENERATOR = None


# --- AGENT STATE DEFINITION (LangGraph) ---
class AgentState(TypedDict):
    question: str
    messages: List[BaseMessage]
    kb_context: List[str]
    web_context: List[str]
    route: str 
    confidence: float
    final_solution: str


# --- DSPy CORE MODULES ---
class MathSolution(dspy.Signature):
    """
    Given the provided context and a student's mathematical question, 
    generate a clear, simplified, step-by-step mathematical solution.
    """
    context = dspy.InputField(desc="Retrieved math content or web text")
    question = dspy.InputField(desc="The student's math question")
    step_by_step_solution = dspy.OutputField(desc="A simplified, numbered, step-by-step solution, clearly stating the final numerical or symbolic answer.")


class DSPyGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(MathSolution)

    def forward(self, context, question):
        try:
            return self.generator(context=context, question=question)
        except Exception as e:
            print(f"DSPy generation error: {e}")
            return type('obj', (object,), {
                'step_by_step_solution': f"Error in generation: {e}",
                'trace': None
            })()


def get_dspy_generator():
    """Lazy initialization of DSPy generator with proper LLM configuration."""
    global DSPY_GENERATOR
    if DSPY_GENERATOR is None:
        try:
            # Use LM class with Ollama endpoint
            dspy_lm = dspy.LM(
                model=f"ollama/{LLM_MODEL}",
                api_base="http://localhost:11434",
                api_key="",
                max_tokens=1000,
                temperature=0.0
            )
            dspy.settings.configure(lm=dspy_lm)
            DSPY_GENERATOR = DSPyGenerator()
            print("✓ DSPy Generator initialized with Ollama")
        except Exception as e:
            print(f"⚠ DSPy initialization failed: {e}")
            DSPY_GENERATOR = None
    return DSPY_GENERATOR

  
# --- UTILITY FUNCTIONS ---
def get_retriever(llm_embeddings=None):
    """Initializes and returns the ChromaDB retriever."""
    global MATH_RETRIEVER
    if MATH_RETRIEVER is None:
        try:
            # Force embeddings to use CPU to avoid GPU memory conflicts
            embeddings_cpu = OllamaEmbeddings(
                model=KB_EMBED_MODEL,
                base_url="http://localhost:11434"
            )
            
            db = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=embeddings_cpu, 
                collection_name=COLLECTION_NAME
            )
            MATH_RETRIEVER = db.as_retriever(search_kwargs={"k": 3})
            print("INFO: ChromaDB Retriever Initialized (CPU mode).")
        except Exception as e:
            print(f"ERROR: Failed to initialize ChromaDB. Is KB setup run? {e}")
            
    return MATH_RETRIEVER


def get_web_search_results(query: str, max_results: int = 3) -> List[str]:
    """Performs DuckDuckGo search and scrapes main content."""
    try:
        search_results = DDGS().text(query, max_results=max_results)
        context = []
        
        for result in search_results:
            url = result.get('href')
            if url and not url.endswith(('.pdf', '.doc')):
                try:
                    response = requests.get(url, timeout=3, headers={'User-Agent': 'MathProfessorAgent/1.0'})
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    main_content = soup.find('body').get_text(separator=' ', strip=True)
                    context.append(main_content[:2500])
                except Exception:
                    continue
        return context
    except Exception as e:
        print(f"Web search error: {e}")
        return []


# --- LANGGRAPH NODE FUNCTIONS ---
def input_guardrail(state: AgentState):
    """Node 1: Checks for MATH topic and safety."""
    print("-> Input Guardrail: Checking Safety/Topic...")
    
    messages = [
        SystemMessage(content="You are an Input Guardrail. Output 'MATH' if the question is mathematical. Output 'NON_MATH' if it is unsafe or non-educational."),
        HumanMessage(content=state['question'])
    ]
    
    response = LLM.invoke(messages).content.strip().upper()
    
    if 'MATH' in response:
        return state
    else:
        return {"route": "REFUSE", "final_solution": "I can only answer educational questions related to mathematics."}


def router_agent(state: AgentState):
    """Node 2: Decides between KB_RAG and WEB_SEARCH."""
    print("-> Routing Agent: Deciding Path...")
    
    messages = [
        SystemMessage(content="Output 'KB_RAG' if the question is standard, textbook-level math. Output 'WEB_SEARCH' if the question is complex, niche, or likely very recent."),
        HumanMessage(content=state['question'])
    ]
    
    response = LLM.invoke(messages).content.strip().upper()

    route = 'KB_RAG' if 'KB_RAG' in response else 'WEB_SEARCH'
    print(f"-> Routing to: {route}")
    return {"route": route}


def kb_retrieval(state: AgentState):
    """Node 3: Retrieves context from ChromaDB."""
    print("-> Retrieving from Knowledge Base...")
    
    retriever = get_retriever()
    
    if retriever is None:
        print("ERROR: Retriever not available. Falling back to Web Search.")
        return {"route": "WEB_SEARCH"} 
        
    docs = retriever.invoke(state['question'])
    kb_context = [doc.page_content for doc in docs]
    print(f"-> Found {len(kb_context)} KB documents.")
    return {"kb_context": kb_context}


def web_search_extraction(state: AgentState):
    """Node 4: Performs web search, extracts content, and includes refusal logic."""
    print("-> Running Web Search/Extraction...")
    web_context = get_web_search_results(state['question'], max_results=3)
    
    if not web_context or len(" ".join(web_context)) < 200:
        print("-> Web search yielded insufficient context. REFUSAL.")
        return {"route": "REFUSE", "web_context": []}
    
    print(f"-> Extracted {len(web_context)} web contexts.")
    return {"web_context": web_context}


def generation_agent(state: AgentState):
    """Node 5: Generates solution using direct LLM (DSPy optional)."""
    print("-> Generating Solution...")
    
    context = state.get('kb_context') or state.get('web_context') or []
    context_str = "\n---\n".join(context)
    
    if not context_str:
        return {"final_solution": "Could not find enough relevant information.", "confidence": 0.0}

    # Use direct LLM call (more reliable than DSPy)
    try:
        messages = [
            SystemMessage(content=f"""You are an expert math tutor. Using the context provided, generate a clear, step-by-step solution to the student's question.

Context:
{context_str}

Instructions:
- Provide numbered steps
- Show all work clearly
- State the final answer explicitly at the end
- Use simple language suitable for students"""),
            HumanMessage(content=state['question'])
        ]
        
        response = LLM.invoke(messages)
        solution_message = response.content
        
        # Check solution quality for confidence scoring
        word_count = len(solution_message.split())
        has_steps = any(str(i) in solution_message for i in range(1, 6))
        has_final_answer = "final answer" in solution_message.lower() or "answer:" in solution_message.lower()
        
        # Calculate confidence
        if word_count > 50 and has_steps and has_final_answer:
            confidence = 0.95
        elif word_count > 30 and has_steps:
            confidence = 0.80
        elif word_count > 20:
            confidence = 0.65
        else:
            confidence = 0.50
        
        print(f"-> Solution generated with confidence: {confidence}")
        return {"final_solution": solution_message, "confidence": confidence}
        
    except Exception as e:
        print(f"LLM Generation FAILED: {e}")
        return {"final_solution": f"Generation failed: {e}", "confidence": 0.0}


def output_guardrail(state: AgentState):
    """Node 6: Checks confidence and flags Human-in-the-Loop."""
    if state['confidence'] < 0.75: 
        print("-> Output Guardrail: LOW CONFIDENCE. Flagging for Human-in-the-Loop.")
        return {"route": "HUMAN_IN_LOOP"}
    else:
        return state


def refuse_handler(state: AgentState):
    """Node 7: Handles final refusal message."""
    return {"final_solution": state.get('final_solution') or "I cannot answer this question safely or accurately.", "confidence": 0.0}


# --- EDGE/CONDITIONAL FUNCTIONS ---
def decide_next_step(state: AgentState) -> str:
    """Routing from Input Guardrail to Router."""
    if state['route'] == 'REFUSE':
        return 'REFUSE_HANDLER'
    return 'ROUTER'


def decide_routing(state: AgentState) -> str:
    """Routing from Router Agent."""
    if state['route'] == 'KB_RAG':
        return 'KB_RETRIEVAL'
    return 'WEB_SEARCH_EXTRACTION'


def decide_kb_or_web(state: AgentState) -> str:
    """Routing from KB_RETRIEVAL (KB success or Fallback)."""
    if state.get('route') == 'WEB_SEARCH':
        return 'WEB_SEARCH_EXTRACTION'
    if state['kb_context']:
        return 'GENERATION'
    return 'WEB_SEARCH_EXTRACTION'


def decide_web_success(state: AgentState) -> str:
    """Routing from WEB_SEARCH_EXTRACTION (Web success or Refusal)."""
    if state['web_context']:
        return 'GENERATION'
    return 'REFUSE_HANDLER'


def decide_final_output(state: AgentState) -> str:
    """Routing from Output Guardrail."""
    if state['route'] == 'HUMAN_IN_LOOP':
        print("-> HITL Flagged. Sending response, but requires manual check.")
        return 'FINAL_OUTPUT'
    return 'FINAL_OUTPUT'


# --- GRAPH CONSTRUCTION ---
def build_math_agent_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("input_guardrail", input_guardrail)
    workflow.add_node("router_agent", router_agent)
    workflow.add_node("kb_retrieval", kb_retrieval)
    workflow.add_node("web_search_extraction", web_search_extraction)
    workflow.add_node("generation_agent", generation_agent)
    workflow.add_node("output_guardrail", output_guardrail)
    workflow.add_node("refuse_handler", refuse_handler) 

    workflow.set_entry_point("input_guardrail")

    workflow.add_conditional_edges(
        "input_guardrail",
        decide_next_step, 
        {'ROUTER': "router_agent", 'REFUSE_HANDLER': "refuse_handler"}
    )

    workflow.add_conditional_edges(
        "router_agent",
        decide_routing,
        {'KB_RETRIEVAL': "kb_retrieval", 'WEB_SEARCH_EXTRACTION': "web_search_extraction"}
    )

    workflow.add_conditional_edges(
        "kb_retrieval",
        decide_kb_or_web,
        {"GENERATION": "generation_agent", "WEB_SEARCH_EXTRACTION": "web_search_extraction"}
    )

    workflow.add_conditional_edges(
        "web_search_extraction",
        decide_web_success,
        {"GENERATION": "generation_agent", "REFUSE_HANDLER": "refuse_handler"}
    )
    
    workflow.add_edge("generation_agent", "output_guardrail")
    
    workflow.add_conditional_edges(
        "output_guardrail",
        decide_final_output,
        {"FINAL_OUTPUT": END, "HUMAN_IN_LOOP": END}
    )
    
    workflow.add_edge("refuse_handler", END)

    return workflow.compile()


# --- FASTAPI APPLICATION ---
def test_ollama_connection():
    """Test if Ollama is running and the model is available."""
    try:
        print(f"Testing Ollama connection with model: {LLM_MODEL}...")
        test_llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        response = test_llm.invoke("Test")
        print(f"✓ Ollama connected successfully!")
        return True
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. ollama list")
        print(f"2. ollama pull {LLM_MODEL}")
        return False


# Test connection on startup
test_ollama_connection()

# Build the workflow
math_agent = build_math_agent_workflow()

# Create FastAPI app
app = FastAPI(title="MathProf API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    prompt: Optional[str] = None
    message: Optional[str] = None
    
    class Config:
        extra = "allow"
    
    def get_question(self) -> str:
        """Get the question from any possible field."""
        return self.question or self.query or self.prompt or self.message or ""


class SolutionResponse(BaseModel):
    question: str
    solution: str
    confidence: float
    route: str


@app.post("/api/ask", response_model=SolutionResponse)
async def ask_question(request: QuestionRequest):
    """Main endpoint for asking math questions."""
    try:
        question_text = request.get_question()
        
        if not question_text:
            raise HTTPException(status_code=400, detail="No question provided")
        
        initial_state = {
            "question": question_text,
            "messages": [],
            "kb_context": [],
            "web_context": [],
            "route": "",
            "confidence": 0.0,
            "final_solution": ""
        }
        
        final_state = math_agent.invoke(initial_state)
        
        return SolutionResponse(
            question=question_text,
            solution=final_state.get("final_solution", "No solution generated"),
            confidence=final_state.get("confidence", 0.0),
            route=final_state.get("route", "unknown")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "MathProf API is running", "version": "1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "ollama_connected": test_ollama_connection()}