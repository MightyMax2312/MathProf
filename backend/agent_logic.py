import json
import requests
import re
from typing import TypedDict, List, Dict, Any
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain_chroma import Chroma                         
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END, START 
from ddgs import DDGS  
from bs4 import BeautifulSoup       
import dspy                         

LLM_MODEL = "granite4:3b"
KB_EMBED_MODEL = "nomic-embed-text" 
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "math_problems"

LLM = ChatOllama(model=LLM_MODEL, temperature=0.0)
embeddings = OllamaEmbeddings(model=KB_EMBED_MODEL)

def clean_math_text(text: str) -> str:
    """Normalize and clean math expressions for proper LaTeX rendering."""
    if not text:
        return text

    text = re.sub(r"<\/?b>", "", text)
    text = re.sub(r"<\/?strong>", "", text)

    text = re.sub(r'([xX])\s*([0-9nN\+\-\(\)]+)', r'\1^{\2}', text)
    text = re.sub(r'([xX])([0-9nN\+\-\(\)]+)', r'\1^{\2}', text)

    text = re.sub(r'∫', r'\\int ', text)
    text = text.replace("=", " = ")
    text = text.replace("dx", " dx")
    text = text.replace("d x", " dx")
    text = re.sub(r'([0-9])\/([0-9])', r'\\frac{\1}{\2}', text)
    text = re.sub(r'\^([0-9nN\+\-\(\)]+)', r'^{\1}', text)
    if not any(tag in text for tag in ["\\(", "\\[", "$"]):
        text = re.sub(r'(\b\d+[\+\-\*/^]\d+\b)', r'\\(\1\\)', text)

    return text

try:
    MATH_RETRIEVER = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings, 
        collection_name=COLLECTION_NAME
    ).as_retriever(search_kwargs={"k": 3})
    print("INFO: ChromaDB Retriever Initialized in agent_logic.py")
except Exception as e:
    print(f"WARNING: Could not initialize retriever in agent_logic.py: {e}")
    MATH_RETRIEVER = None


class AgentState(TypedDict):
    question: str
    messages: List[BaseMessage]
    kb_context: List[str]
    web_context: List[str]
    route: str 
    confidence: float
    final_solution: str


class MathSolution(dspy.Signature):
    context = dspy.InputField(desc="Retrieved math content or web text")
    question = dspy.InputField(desc="The student's math question")
    step_by_step_solution = dspy.OutputField(desc="Simplified, step-by-step solution with final answer.")


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


DSPY_GENERATOR = None


def initialize_dspy():
    """Initialize DSPy with Ollama configuration."""
    global DSPY_GENERATOR
    if DSPY_GENERATOR is None:
        try:
            dspy_lm = dspy.LM(
                model=f"ollama/{LLM_MODEL}",
                api_base="http://localhost:11434",
                api_key="",
                max_tokens=1000,
                temperature=0.0
            )
            dspy.settings.configure(lm=dspy_lm)
            DSPY_GENERATOR = DSPyGenerator()
            print("✓ DSPy Generator initialized")
            return True
        except Exception as e:
            print(f"⚠ DSPy initialization failed: {e}")
            return False
    return True


def get_web_search_results(query: str, max_results: int = 3) -> List[str]:
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


def input_guardrail(state: AgentState):
    print("-> Input Guardrail: Checking Safety/Topic...")
    messages = [
        SystemMessage(content="You are an Input Guardrail. Output 'MATH' if the question is mathematical. Output 'NON_MATH' if unsafe or unrelated."),
        HumanMessage(content=state['question'])
    ]
    response = LLM.invoke(messages).content.strip().upper()
    if 'MATH' in response:
        return state
    return {"route": "REFUSE", "final_solution": "I can only answer educational questions related to mathematics."}


def router_agent(state: AgentState):
    print("-> Routing Agent: Deciding Path...")
    messages = [
        SystemMessage(content="Output 'KB_RAG' if textbook-level math; 'WEB_SEARCH' if complex or recent."),
        HumanMessage(content=state['question'])
    ]
    response = LLM.invoke(messages).content.strip().upper()
    route = 'KB_RAG' if 'KB_RAG' in response else 'WEB_SEARCH'
    print(f"-> Routing to: {route}")
    return {"route": route}


def kb_retrieval(state: AgentState):
    print("-> Retrieving from Knowledge Base...")
    if MATH_RETRIEVER is None:
        print("ERROR: Retriever not available. Falling back to Web Search.")
        return {"route": "WEB_SEARCH", "kb_context": []}
    try:
        docs = MATH_RETRIEVER.invoke(state['question'])
        kb_context = [doc.page_content for doc in docs]
        print(f"-> Found {len(kb_context)} KB documents.")
        return {"kb_context": kb_context}
    except Exception as e:
        print(f"ERROR during retrieval: {e}. Falling back to Web Search.")
        return {"route": "WEB_SEARCH", "kb_context": []}


def web_search_extraction(state: AgentState):
    print("-> Running Web Search/Extraction...")
    web_context = get_web_search_results(state['question'], max_results=3)
    if not web_context or len(" ".join(web_context)) < 200:
        print("-> Web search insufficient. REFUSAL.")
        return {"route": "REFUSE", "web_context": []}
    print(f"-> Extracted {len(web_context)} web contexts.")
    return {"web_context": web_context}


def generation_agent(state: AgentState):
    print("-> Generating Solution...")
    context = state.get('kb_context') or state.get('web_context') or []
    context_str = "\n---\n".join(context)

    if not context_str:
        return {"final_solution": "Could not find enough relevant information.", "confidence": 0.0}

    try:
        messages = [
            SystemMessage(content=f"""You are an expert math tutor. Using the context provided, generate a clear, step-by-step solution to the student's question.

Context:
{context_str}

Instructions:
- Provide numbered steps
- Show all work clearly
- State the final answer explicitly at the end
- Use LaTeX-style math formatting like \\(x^2\\), \\(\\int x dx\\)
- Use simple student-friendly language"""),
            HumanMessage(content=state['question'])
        ]
        response = LLM.invoke(messages)
        solution_message = clean_math_text(response.content)

        word_count = len(solution_message.split())
        has_steps = any(str(i) in solution_message for i in range(1, 6))
        has_final_answer = "final answer" in solution_message.lower() or "answer:" in solution_message.lower()

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
    if state['confidence'] < 0.75: 
        print("-> Output Guardrail: LOW CONFIDENCE. HITL triggered.")
        return {"route": "HUMAN_IN_LOOP"}
    return state


def refuse_handler(state: AgentState):
    return {"final_solution": state.get('final_solution') or "I cannot answer this question safely or accurately.", "confidence": 0.0}


def decide_next_step(state: AgentState) -> str:
    return 'REFUSE_HANDLER' if state['route'] == 'REFUSE' else 'ROUTER'


def decide_routing(state: AgentState) -> str:
    return 'KB_RETRIEVAL' if state['route'] == 'KB_RAG' else 'WEB_SEARCH_EXTRACTION'


def decide_kb_or_web(state: AgentState) -> str:
    if state.get('route') == 'WEB_SEARCH':
        return 'WEB_SEARCH_EXTRACTION'
    return 'GENERATION' if state.get('kb_context') else 'WEB_SEARCH_EXTRACTION'


def decide_web_success(state: AgentState) -> str:
    return 'GENERATION' if state.get('web_context') else 'REFUSE_HANDLER'


def decide_final_output(state: AgentState) -> str:
    if state.get('route') == 'HUMAN_IN_LOOP':
        print("-> HITL Flagged. Sending response, but requires manual check.")
    return END


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

    workflow.add_conditional_edges("input_guardrail", decide_next_step,
        {'ROUTER': "router_agent", 'REFUSE_HANDLER': "refuse_handler"})
    workflow.add_conditional_edges("router_agent", decide_routing,
        {'KB_RETRIEVAL': "kb_retrieval", 'WEB_SEARCH_EXTRACTION': "web_search_extraction"})
    workflow.add_conditional_edges("kb_retrieval", decide_kb_or_web,
        {"GENERATION": "generation_agent", "WEB_SEARCH_EXTRACTION": "web_search_extraction"})
    workflow.add_conditional_edges("web_search_extraction", decide_web_success,
        {"GENERATION": "generation_agent", "REFUSE_HANDLER": "refuse_handler"})
    workflow.add_edge("generation_agent", "output_guardrail")
    workflow.add_edge("output_guardrail", END)
    workflow.add_edge("refuse_handler", END)
    return workflow.compile()


if __name__ == "__main__":
    print("Building Math Agent Workflow...")
    agent = build_math_agent_workflow()

    test_question = "Integrate x^n with respect to x"
    print(f"\nTesting with question: {test_question}")

    initial_state = {
        "question": test_question,
        "messages": [],
        "kb_context": [],
        "web_context": [],
        "route": "",
        "confidence": 0.0,
        "final_solution": ""
    }

    result = agent.invoke(initial_state)
    print(f"\n--- RESULT ---")
    print(f"Solution: {result['final_solution']}")
    print(f"Confidence: {result['confidence']}")
