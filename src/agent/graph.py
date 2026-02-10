import operator
from typing import Annotated, List, TypedDict, Union, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.redis import RedisSaver
except ImportError:
    RedisSaver = None

from src.core.logger import get_logger
from src.core.history import redis_client
from src.agent.planner import create_retrieval_planner, RetrievalPlan
from src.rag.retrieval import BM25SemanticHybridRetriever, Neo4jVectorRetriever
from src.prompts.templates import template_chat

log = get_logger("agent_graph")

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    input: str
    messages: Annotated[List[BaseMessage], add_messages]  # Use add_messages reducer for automatic appending
    selected_files: List[str] # Filter
    
    # Internal state
    plan: dict # Serialized dictionary (was RetrievalPlan)
    # context: Annotated[List[Document], operator.add] # REMOVED: Caused accumulation and serialization issues
    hybrid_docs: List[dict] # Serialized docs, overwrites per turn
    graph_docs: List[dict] # Serialized docs, overwrites per turn
    answer: str



def create_agentic_rag_graph(
    llm_chat,
    hybrid_retriever: BM25SemanticHybridRetriever, 
    neo4j_retriever: Neo4jVectorRetriever = None,
    checkpointer = None
):
    """Creates the Agentic RAG Graph using LangGraph.
    
    Args:
        checkpointer: LangGraph checkpointer for conversation memory (e.g., MemorySaver)
    """
    
    # 1. Nodes
    
    
    def planner_node(state: AgentState):
        """Decides which retrieval path to take."""
        print("[DEBUG GRAPH] ===== PLANNER NODE CALLED =====")
        print(f"[DEBUG GRAPH] State keys: {state.keys()}")
        print(f"[DEBUG GRAPH] Input: {state.get('input', 'NO INPUT')[:100]}")
        
        from config import settings
        from src.agent.planner import get_document_summaries # Import
        
        log.info("Agent: Planning retrieval strategy...")
        
        # Fetch summaries for context
        summaries_context = "No summaries available."
        try:
            # Extract clients from hybrid retriever (hacky but effective reuse)
            q_client = hybrid_retriever.qdrant_client
            embs = hybrid_retriever.embeddings
            
            if q_client and embs:
                summaries_context = get_document_summaries(
                    query=state["input"],
                    qdrant_client=q_client,
                    embeddings=embs
                )
        except Exception as e:
            log.warning(f"Could not fetch summaries for planner: {e}")

        planner = create_retrieval_planner(llm_model=settings.LLM_CHAT_MODEL_NAME) # Use configured model
        plan = planner.invoke({
            "query": state["input"],
            "summaries": summaries_context
        })
        log.info(f"Agent: Plan selected: {plan.strategy} (Reasoning: {plan.reasoning})")
        return {"plan": plan.dict()}

    def hybrid_search_node(state: AgentState):
        """Executes Hybrid (BM25+Vector) Search."""
        plan = state["plan"]
        if "search_queries" in plan and not plan["search_queries"]:
            log.info("Agent: Planner returned no search queries. Skipping Hybrid Search.")
            return {"hybrid_docs": []}

        log.info("Agent: Executing Hybrid Search...")
        # We use state["input"] for search context as it works well, 
        # but the check above ensures we respect the planner's decision to SKIP.
        docs = hybrid_retriever.retrieve_hybrid(
            state["input"], 
            selected_files=state.get("selected_files")
        )
        # Serialize to dicts
        return {"hybrid_docs": [d.dict() for d in docs]}

    def graph_search_node(state: AgentState):
        """Executes Graph Vector Search."""
        if not neo4j_retriever:
            log.warning("Agent: Graph retriever requested but not available. Skiping.")
            return {"graph_docs": []}
            
        log.info("Agent: Executing Graph Search...")
        docs = neo4j_retriever.get_relevant_documents(
            state["input"],
            selected_files=state.get("selected_files")
        )
        # Serialize to dicts
        return {"graph_docs": [d.dict() for d in docs]}

    def generate_node(state: AgentState):
        """Generates the final answer."""
        print("[DEBUG GRAPH] ===== GENERATOR NODE CALLED =====")
        
        # Combine docs from hybrid and graph (lists of dicts)
        hybrid_docs = state.get("hybrid_docs", [])
        graph_docs = state.get("graph_docs", [])
        
        all_docs = hybrid_docs + graph_docs
        unique_docs = []
        seen = set()
        
        for d in all_docs:
            content = d.get('page_content', '')
            if content and content not in seen:
                unique_docs.append(d)
                seen.add(content)
        
        # DEBUG: Check context
        print(f"[DEBUG GRAPH] Context docs count: {len(unique_docs)}")
        if unique_docs:
            print(f"[DEBUG GRAPH] First doc preview: {unique_docs[0].get('page_content', '')[:200]}")
        else:
            print("[DEBUG GRAPH] WARNING: NO CONTEXT DOCUMENTS!")
        
        # DEBUG: Check chat_history state
        chat_history = state.get("messages", [])
        print(f"[DEBUG GRAPH] Chat history length: {len(chat_history)}")
        if chat_history:
            print(f"[DEBUG GRAPH] First 2 history messages:")
            for i, msg in enumerate(chat_history[:2]):
                print(f"[DEBUG GRAPH]   {i}: {type(msg).__name__}: {str(msg)[:150]}")
        
        log.info("Agent: Generating Answer...")
        
        # Format context
        # CRITICAL: If no docs, keep empty so prompt uses history. 
        # "No documents found" triggers the "Ignore History" rule in some models.
        context_str = "\n\n".join([d.get('page_content', '') for d in unique_docs]) if unique_docs else ""
        
        chain = template_chat | llm_chat
        
        # Prepare input
        chain_input = {
            "context": context_str,
            "messages": chat_history,
            "input": state["input"]
        }
        
        try:
            response = chain.invoke(chain_input)
            log.info(f"[DEBUG] Generator Node - Response generated: {response.content[:100]}...")
            
            # Return response
            return {
                "answer": response.content,
                "messages": [response]
            }
        except Exception as e:
            log.error(f"Error in generate_node during chain invocation: {e}")
            return {"answer": "Error generating answer."}

    # 2. Graph Construction
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("hybrid_search", hybrid_search_node)
    workflow.add_node("graph_search", graph_search_node)
    workflow.add_node("generator", generate_node)
    
    # Entry
    workflow.set_entry_point("planner")
    
    # Conditional Edges for Retrieval
    def route_retrieval(state: AgentState):
        strategy = state["plan"]["strategy"]
        if strategy == "hybrid":
            return ["hybrid_search"]
        elif strategy == "graph":
            return ["graph_search"]
        elif strategy == "both":
            return ["hybrid_search", "graph_search"]
        return ["hybrid_search"] # Fallback

    workflow.add_conditional_edges(
        "planner",
        route_retrieval,
        {
            "hybrid_search": "hybrid_search",
            "graph_search": "graph_search"
        } 
    )
    
    # Edges to Generator
    workflow.add_edge("hybrid_search", "generator")
    workflow.add_edge("graph_search", "generator")
    
    workflow.add_edge("generator", END)
    
    # Compile with checkpointer for conversation memory
    if checkpointer is None:
        try:
            from langgraph.checkpoint.redis.asyncio import AsyncRedisSaver
            from config import settings
            
            # Async checkpointer is required for async FastAPI apps
            redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
            checkpointer = AsyncRedisSaver.from_conn_string(redis_url)
            print(f"!!! [DEBUG] Initialized AsyncRedisSaver checkpointer: {redis_url} !!!")
            log.info("Initialized AsyncRedisSaver checkpointer for LangGraph")
        except Exception as e:
            print(f"!!! [DEBUG] Failed to initialize AsyncRedisSaver: {e} !!!")
            log.warning(f"Failed to initialize AsyncRedisSaver: {e}. Falling back to MemorySaver.")
            checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)
