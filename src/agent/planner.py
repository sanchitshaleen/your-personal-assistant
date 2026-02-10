import os
from typing import TypedDict, Literal, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from src.core.logger import get_logger
from src.core.llm import get_llm
from config import settings

log = get_logger("retrieval_planner")

class RetrievalPlan(BaseModel):
    """Plan for retrieval strategy."""
    strategy: Literal["hybrid", "graph", "both"] = Field(
        description="The retrieval strategy to use. 'hybrid' for general text search, 'graph' for enhancing complex relationship queries, 'both' if unsure."
    )
    reasoning: str = Field(description="Reasoning for the chosen strategy.")
    search_queries: List[str] = Field(description="List of search queries to use for the chosen strategy.")

def create_retrieval_planner(llm_model: str = "gemma2:9b"):
    """Creates a planner chain that decides which retrieval strategy to use."""
    llm = get_llm(model_name=llm_model, context_size=8192, temperature=0.0) # Use a smarter model for planning if possible
    
    # Structured output for reliability
    structured_llm = llm.with_structured_output(RetrievalPlan)
    
    system_prompt = """You are an expert Retrieval Planner for a RAG system.
Your goal is to analyze the user's query and decide the best retrieval strategy.

Available Strategies:
1. "hybrid": Uses BM25 (keyword) + Semantic (vector) search. Best for finding specific facts, looking up attributes (e.g., skills from a resume), or finding specific documents (e.g., "Find the contract").
2. "graph": Uses Knowledge Graph vector search. Best for deep conceptual questions, detailed explanations, and specifically for comparisons or differences between entities (e.g., "Compare X and Y", "How do X and Y differ?").
3. "both": Uses both strategies. Use this if the query is ambiguous, requires coverage of multiple document types, or if you are unsure.

Context:
You may have access to document summaries. Use them to infer if the available documents are suitable for the user's query.

Instructions:
- Analyze the user's query and the provided "Document Summaries".
- If the summaries indicate documents that contain specific facts (e.g., "Resume", "Contract", "List of..."), choose "hybrid".
- If the summaries indicate documents that contain complex knowledge, concepts, or entities (e.g., "Interview Questions", "Technical Guides", "Research Papers"), choose "graph" to leverage relationship traversal.
- Default to "hybrid" if the query is simple factual lookup.
- Prefer "graph" if the query implicates a relationship or comparison between two technical concepts.
- SPECIAL CASE: If the user query is asking about the CONVERSATION HISTORY (e.g., "what did I ask", "previous question", "summarize our chat"), then:
    - Choose "hybrid" strategy.
    - Set "search_queries" to an EMPTY list []. (This ensures no documents are retrieved, forcing the system to read from History).
- Generate 1-3 optimized search queries (unless Special Case).
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Query: {query}\n\nDocument Summaries (Context):\n{summaries}")
    ])
    
    planner_chain = prompt | structured_llm
    return planner_chain

def get_document_summaries(query: str, qdrant_client, embeddings, collection_name="document_summaries", k=5) -> str:
    """
    Fetches relevant document summaries from Qdrant to aid planning.
    """
    try:
        # 1. Embed query
        query_vector = embeddings.embed_query(query)
        
        # 2. Search Qdrant
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=k
        )
        
        if not results:
            return "No relevant document summaries found."
            
        # 3. Check ingestion targets
        ingestion_targets = set()
        for res in results:
            target = res.payload.get('ingestion_target', 'qdrant')
            ingestion_targets.add(target)
        
        # 4. Format results
        summaries = []
        for res in results:
            payload = res.payload
            filename = payload.get('filename', 'unknown')
            content = payload.get('page_content', '')
            target = payload.get('ingestion_target', 'qdrant')
            summaries.append(f"File: {filename}\nStorage: {target.upper()}\nSummary: {content}")
        
        # 5. Add storage location guidance
        result_text = "\n\n".join(summaries)
        if ingestion_targets == {'neo4j'}:
            result_text += "\n\n**IMPORTANT: All relevant documents are stored in NEO4J (Knowledge Graph database). You MUST use the 'graph' strategy to retrieve them.**"
        elif 'neo4j' in ingestion_targets and 'qdrant' in ingestion_targets:
            result_text += "\n\n**IMPORTANT: Documents are stored in both NEO4J and QDRANT. Consider using 'both' strategy for comprehensive coverage.**"
        
        return result_text
        
    except Exception as e:
        log.error(f"Failed to fetch document summaries: {e}")
        return "Error fetching summaries."
