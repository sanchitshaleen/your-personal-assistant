# FastAPI server which will handle all the backend and GenAI aspects of the application
# uvicorn server:app --reload
# Avoid using --reload flag, because, LLMs will keep reloading and system will overheat.

from fastapi import FastAPI, File, UploadFile, Form, Request, Query, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import json
import time
import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

from src.core.logger import get_logger
log = get_logger("main")
log.info("🚀 RAG Service Initialized with Agentic Planner Integration")

# llm system imports:
from src.core.llm import get_llm, get_output_parser
from src.core.llm import get_dummy_response
from src.core.llm import get_dummy_response_stream
from src.rag.vector_store import VectorDB
from langchain_core.messages import HumanMessage
from config import settings as config
from config import settings as config
from src.core.history import HistoryStore, redis_client  # Class
from src.rag.cache import SemanticCache    # Class
from src.core.cache_version import get_cache_version_manager  # Function
from src.prompts.chains import build_rag_chain           # Function
from config import settings as config                               # Constants
from src.rag.indexer import ingest_file           # Function

# Celery imports for async background tasks
from src.worker import ingest_file_task

# Helper Modules:
from src.core import database as pg_db
from src.processing import metadata as metadata_extractor
from src.processing import files as files

# Type hinting imports:
from langchain_core.vectorstores import VectorStore as T_VECTOR_STORE
from langchain_core.messages import BaseMessage as T_MESSAGE
from langchain_core.runnables import Runnable

from src.core import logger
log = logger.get_logger("rag_server")


# Helper function to detect if query is about conversation history using LLM
async def is_conversation_history_question(query: str, llm) -> bool:
    """Use keyword matching to detect if the query is asking about previous messages or conversation history."""
    query_lower = query.lower()
    
    # Keywords that indicate conversation history questions
    history_keywords = [
        "what did i ask",
        "my last question",
        "my first question",
        "my previous question",
        "what was my",
        "earlier in",
        "conversation history",
        "what did i say",
        "my earlier",
        "that question",
        "first message",
        "last message",
        "previous message",
        "my 1st",
        "my 2nd",
        "my second",
        "my third",
        "my 3rd"
    ]
    
    # Check if any keyword is in the query
    for keyword in history_keywords:
        if keyword in query_lower:
            log.info(f"[DETECT] Query: '{query}' | Keyword matched: '{keyword}' | Detected: True")
            return True
    
    log.info(f"[DETECT] Query: '{query}' | No keywords matched | Detected: False")
    return False


def get_conversation_answer(history_messages: list, query: str) -> str:
    """Generate an answer based on conversation history without using RAG."""
    # Ensure we have properly parsed messages
    parsed_messages = []
    for msg in history_messages:
        if isinstance(msg, dict):
            parsed_messages.append(msg)
        elif isinstance(msg, str):
            try:
                parsed_messages.append(json.loads(msg))
            except:
                pass
    
    # Filter to only human messages
    human_messages = [msg for msg in parsed_messages if isinstance(msg, dict) and msg.get("role") == "human"]
    
    if not human_messages:
        return "I don't have any previous questions in our conversation history."
    
    query_lower = query.lower()
    
    # Try to identify which message they're asking about
    if "last" in query_lower or "most recent" in query_lower or "recent" in query_lower:
        last_question = human_messages[-1].get("content", "")
        return f"Your last question was: \"{last_question}\""
    
    elif "first" in query_lower or "1st" in query_lower or "beginning" in query_lower or "start" in query_lower:
        first_question = human_messages[0].get("content", "")
        return f"Your first question was: \"{first_question}\""
    
    elif "second" in query_lower or "2nd" in query_lower:
        if len(human_messages) > 1:
            second_question = human_messages[1].get("content", "")
            return f"Your second question was: \"{second_question}\""
        else:
            return "You don't have a second question in our conversation."
    
    elif "previous" in query_lower or "before" in query_lower or "prior" in query_lower:
        if len(human_messages) > 1:
            prev_question = human_messages[-2].get("content", "")
            return f"Your previous question was: \"{prev_question}\""
        else:
            return "There are no previous questions before this one."
    
    else:
        # Generic conversation history response - list all questions
        questions = "\n".join([f"• {msg.get('content', '')}" for msg in human_messages])
        return f"Here are all your questions in this conversation:\n{questions}"


# ------------------------------------------------------------------------------
# Constants:
# ------------------------------------------------------------------------------

# UPLOADS_DIR: str = "user_uploads"
OLD_FILE_THRESHOLD: int = 3600 * 1  # 24 hours in seconds
# OLD_FILE_THRESHOLD: int = 20         # 1 min


# ------------------------------------------------------------------------------
# FastAPI Startup:
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define the lifespan context manager for startup/shutdown"""

    # [ Startup ]
    log.info("[LifeSpan] Starting the server components.")
    
    # Run health checks before initialization
    log.info("[LifeSpan] Running health checks...")
    try:
        from src.core.health import run_health_checks
        if not run_health_checks(verbose=True):
            log.warning("[LifeSpan] ⚠️  Some health checks failed. Continuing anyway...")
    except Exception as e:
        log.warning(f"[LifeSpan] Health check initialization failed: {e}. Continuing anyway...")
    
    # Initialize dynamic configuration based on model capabilities
    log.info("[LifeSpan] Initializing dynamic configuration...")
    config._initialize_dynamic_config()
    log.info(f"[LifeSpan] Config initialized - MAX_CONTENT_SIZE: {config.MAX_CONTENT_SIZE}, "
             f"DOC_CHAR_LIMIT: {config.DOC_CHAR_LIMIT}, DOCS_NUM_COUNT: {config.DOCS_NUM_COUNT}")

    app.state.llm_chat = get_llm(
        model_name=config.LLM_CHAT_MODEL_NAME,
        context_size=config.MAX_CONTENT_SIZE,
        temperature=config.LLM_CHAT_TEMPERATURE,
        verify_connection=config.VERIFY_LLM_CONNECTION
    )

    # app.state.llm_summary = get_llm(...)
    app.state.llm_summary = app.state.llm_chat

    app.state.output_parser = get_output_parser()
    app.state.vector_db = VectorDB(
        embed_model=config.EMB_MODEL_NAME,
        retriever_num_docs=config.DOCS_NUM_COUNT,
        verify_connection=config.VERIFY_EMB_CONNECTION,
        qdrant_host=config.QDRANT_HOST,
        qdrant_port=config.QDRANT_PORT,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    app.state.history_store = HistoryStore()

    # Initialize retriever (BM25 hybrid or standard semantic)
    # Initialize retriever (BM25 hybrid or standard semantic)
    # Always init Hybrid to pass to Graph
    log.info("Initializing Hybrid Retrieval for Agent...")
    hybrid_retriever = app.state.vector_db.get_bm25_semantic_hybrid_retriever(
        bm25_weight=config.BM25_WEIGHT,
        semantic_weight=config.SEMANTIC_WEIGHT
    )
    # Init Neo4j if enabled
    neo4j_retriever = app.state.vector_db.get_neo4j_retriever(k=config.DOCS_NUM_COUNT)
    app.state.neo4j_retriever = neo4j_retriever # Store for direct access
    
    app.state.retriever = hybrid_retriever # For compatibility if needed

    # Build Agentic RAG Graph
    log.info("Building Agentic RAG Graph...")
    from src.agent.graph import create_agentic_rag_graph
    
    agent_graph = None
    try:
        from langgraph.checkpoint.redis.asyncio import AsyncRedisSaver
        redis_url = f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}"
        os.environ["REDIS_URL"] = redis_url # Critical for redisvl used internally by LangGraph
        checkpointer = AsyncRedisSaver.from_conn_string(redis_url)
        # Async setup is crucial for connection pool initialization
        if hasattr(checkpointer, "asetup"):
            await checkpointer.asetup()
        log.info(f"✅ Initialized AsyncRedisSaver for Graph Memory: {redis_url}")
        
        agent_graph = create_agentic_rag_graph(
            llm_chat=app.state.llm_chat,
            hybrid_retriever=hybrid_retriever,
            neo4j_retriever=neo4j_retriever,
            checkpointer=checkpointer
        )
    except Exception as e:
        log.warning(f"⚠️ Failed to init AsyncRedisSaver: {e}. Graph will use fallback MemorySaver.")
        # Fallback
        agent_graph = create_agentic_rag_graph(
            llm_chat=app.state.llm_chat,
            hybrid_retriever=hybrid_retriever,
            neo4j_retriever=neo4j_retriever
        )
    
    # Use agent graph directly - the checkpointer handles memory automatically!
    # No need for RunnableWithMessageHistory wrapper with LangGraph
    app.state.rag_chain = agent_graph

    # Initialize cache version manager for invalidation detection
    cache_version_mgr = get_cache_version_manager()
    app.state.cache_version = cache_version_mgr.get_version()
    app.state.cache_version_prefix = cache_version_mgr.get_cache_prefix()

    # Initialize semantic cache if enabled
    if config.SEMANTIC_CACHE_ENABLED:
        if redis_client is not None:
            log.info(
                f"Initializing semantic cache "
                f"(version={app.state.cache_version}, "
                f"threshold={config.SEMANTIC_CACHE_SIMILARITY_THRESHOLD}, "
                f"ttl={config.SEMANTIC_CACHE_TTL_SECONDS}s)..."
            )
            try:
                app.state.semantic_cache = SemanticCache(
                    embeddings_model=app.state.vector_db.embeddings,
                    redis_client=redis_client,
                    similarity_threshold=config.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
                    version_hash=app.state.cache_version,
                )
                log.info("✓ Semantic cache initialized successfully")
            except Exception as e:
                log.error(f"✗ Failed to initialize semantic cache: {e}")
                app.state.semantic_cache = None
        else:
            log.warning("✗ Semantic caching enabled but Redis is unavailable - caching disabled")
            app.state.semantic_cache = None
    else:
        log.info("Semantic caching disabled")
        app.state.semantic_cache = None

    log.info("[LifeSpan] All LLM components initialized.")

    # pg_db.delete_database()
    pg_db.create_tables()

    # Files
    files.check_create_uploads_folder()
    files.delete_empty_user_folders()

    # [ Lifespan ]
    yield

    # [ Shutdown ]
    log.info("[LifeSpan] Shutting down LLM server...")
    # Add any cleanup part here
    # Like saving vector DB, or shutting down subprocesses


# Make one FastAPI app instance with the lifespan context manager
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8501",
        "http://127.0.0.1:5500",
        # "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)


# ------------------------------------------------------------------------------
# Basic API Endpoints:
# ------------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint to check if the server is running."""
    return {
        "message": "LLM RAG Server is running!",
        "further": "Proceed to code ur application :)",
        "thought": "You really are not supposed to be reading this waste of time, but if you are, then you are a curious person. I like that! 😄",
    }


# Define data model for chat request
class BasicChatRequest(BaseModel):
    query: str
    session_id: str
    dummy: bool = False


@app.post("/simple")
async def simple(request: Request, chat_request: BasicChatRequest):
    """Endpoint to handle ont time generation queries.
    - Post request expects JSON `{"query": "", "session_id": "", "dummy":T/F}` structure.
    - Return JSON with `{"response": "", "session_id": ""}` structure.
    """

    llm = request.app.state.llm_chat | request.app.state.output_parser
    session_id = chat_request.session_id.strip() or "unknown_session"

    try:
        query = chat_request.query
        dummy = chat_request.dummy
        log.info(f"/simple Requested by '{session_id}'")

        if dummy:
            log.info(f"/simple Dummy response returned for '{session_id}'")
            return get_dummy_response()

        else:
            result = await llm.ainvoke(input=query)

            log.info(f"/simple Response generated for '{session_id}'.")
            return {"response": result, "session_id": session_id}

    except Exception as e:

        log.exception(f"/simple Error {e} for '{session_id}'")
        return JSONResponse(status_code=500, content={"error": str(e)})


# Make one streaming endpoint for the Simple LLM response:
class StreamChatRequest(BaseModel):
    query: str
    session_id: str
    dummy: bool = False


@app.post("/simple/stream")
async def chat_stream(request: Request, chat_request: StreamChatRequest):
    """Endpoint to handle streaming responses for one time generation queries.
    - Post request expects JSON `{"query": "", "session_id": "", "dummy":T/F}` structure.
    - Return NDJSON with types "metadata", "content", or "error".
    """
    llm = request.app.state.llm_chat | request.app.state.output_parser
    session_id = chat_request.session_id.strip() or "unknown_session"

    async def token_streamer():
        try:
            dummy = chat_request.dummy
            s = 'dummy' if dummy else 'real'
            log.info(f"/simple/stream {s} response requested by '{session_id}'")

            # Start be sending meta data first.
            yield json.dumps({
                "type": "metadata",
                "data": {"session_id": session_id}
            }) + "\n"
            # NDJSON (newline-delimited JSON) - Frontend will merge full response my splitting this

            #  Then send the actual response content:
            if dummy:
                # If dummy is True, stream dummy response
                resp = get_dummy_response_stream(
                    batch_tokens=config.BATCH_TOKEN_PS,
                    token_rate=config.TOKENS_PER_SEC
                )
                for chunk in resp:
                    if await request.is_disconnected():
                        log.warning(f"/simple/stream client disconnected for '{session_id}'")
                        break

                    yield json.dumps({
                        "type": "content",
                        "data": chunk
                    }) + "\n"

            else:
                async for chunk in llm.astream(chat_request.query):
                    if await request.is_disconnected():
                        log.warning(f"/simple/stream client disconnected for '{session_id}'")
                        break

                    yield json.dumps({
                        "type": "content",
                        "data": chunk
                    }) + "\n"

            # In the end, you can send some "Done" etc if u need some conditional logic
            # Server will auto send EOF to mark end of generator response.
            # yield json.dumps({
            #     "type": "end",
            #     "data": "done"
            # }) + "\n"
            log.info(f"/simple/stream Streaming completed for '{session_id}'")

        except Exception as e:
            log.exception(f"/simple/stream Error {e} for '{session_id}'")
            yield json.dumps({
                "type": "error",
                "data": str(e)
            }) + "\n"

    # Return a StreamingResponse with the token streamer generator (basically enable streaming)
    return StreamingResponse(token_streamer(), media_type="text/plain")


# RAG endpoint for context-aware chat with document retrieval
class RAGChatRequest(BaseModel):
    query: str
    session_id: str
    selected_files: list[str] = []
    dummy: bool = False


@app.post("/rag")
async def rag_chat(request: Request, chat_request: RAGChatRequest):
    """Endpoint for RAG-based chat with document context.
    - Post request expects JSON with query, session_id, selected_files (optional), and dummy flag.
    - Returns streaming NDJSON with types "metadata", "content", "sources", or "error".
    """
    session_id = chat_request.session_id.strip() or "default_user"
    
    def is_greeting(query: str) -> bool:
        greetings = [
            "hi", "hello", "hey", "how are you", "good morning", "good afternoon", "good evening",
            "greetings", "what's up", "yo", "sup", "howdy", "hola", "namaste", "bonjour", "hallo", "ciao"
        ]
        q = query.strip().lower()
        return any(q == g or q.startswith(g + ' ') or q.endswith(' ' + g) or q in g for g in greetings)

    def is_general_knowledge_question(query: str) -> bool:
        # Add more patterns as needed for your use case
        general_patterns = [
            "weather", "temperature", "news", "sports", "score", "stock", "market", "currency", "exchange rate",
            "president", "prime minister", "capital of", "population of", "time in", "date today", "today's date",
            "who is", "what is", "when is", "where is", "how is", "tell me about", "latest", "current"
        ]
        q = query.strip().lower()
        # If the query contains any of these patterns and does not mention uploaded files, treat as general
        return any(p in q for p in general_patterns)

    async def rag_streamer():
        import time
        start_time = time.time()
        
        try:
            query = chat_request.query
            selected_files = chat_request.selected_files
            dummy = chat_request.dummy

            s = 'dummy' if dummy else 'real'
            log.info(f"/rag {s} response requested by '{session_id}' with files: {selected_files}")

            # Send metadata first
            yield json.dumps({
                "type": "metadata",
                "data": {
                    "session_id": session_id,
                    "selected_files": selected_files
                }
            }) + "\n"

            # Handle greetings/small talk directly
            if is_greeting(query):
                greeting_response = "Hello! How can I help you today?"
                for char in greeting_response:
                    if await request.is_disconnected():
                        log.warning(f"/rag client disconnected for '{session_id}' (greeting)")
                        break
                    yield json.dumps({
                        "type": "content",
                        "data": char
                    }) + "\n"
                    await asyncio.sleep(0.01)
                # Add to history
                history_store: HistoryStore = request.app.state.history_store
                history_store.add_interaction(session_id, query, greeting_response)
                log.info(f"/rag Greeting response completed for '{session_id}'")
                return

            # Handle general knowledge/open domain questions
            if is_general_knowledge_question(query):
                polite_response = (
                    "Sorry, I don’t have access to real-time or general world information. "
                    "I can only answer questions based on the documents you’ve uploaded. "
                    "Please ask something related to your uploaded files."
                )
                for char in polite_response:
                    if await request.is_disconnected():
                        log.warning(f"/rag client disconnected for '{session_id}' (general knowledge)")
                        break
                    yield json.dumps({
                        "type": "content",
                        "data": char
                    }) + "\n"
                    await asyncio.sleep(0.01)
                # Add to history
                history_store: HistoryStore = request.app.state.history_store
                history_store.add_interaction(session_id, query, polite_response)
                log.info(f"/rag General knowledge response completed for '{session_id}'")
                return

            if dummy:
                # Dummy response
                dummy_text = "This is a dummy RAG response for testing."
                for char in dummy_text:
                    yield json.dumps({
                        "type": "content",
                        "data": char
                    }) + "\n"
                    await asyncio.sleep(0.05)

                yield json.dumps({
                    "type": "sources",
                    "data": {"files": selected_files if selected_files else []}
                }) + "\n"

                log.info(f"/rag Dummy streaming completed for '{session_id}'")
                return

            # Check if the query is about conversation history
            llm_check = request.app.state.llm_chat
            is_history_query = await is_conversation_history_question(query, llm_check)

            if is_history_query:
                # Handle conversation history query without RAG
                log.info(f"/rag Detected conversation history question for '{session_id}'")
                history_store: HistoryStore = request.app.state.history_store
                history_messages = history_store.get_history(session_id)
                answer = get_conversation_answer(history_messages, query)

                for char in answer:
                    if await request.is_disconnected():
                        log.warning(f"/rag client disconnected for '{session_id}'")
                        break
                    yield json.dumps({
                        "type": "content",
                        "data": char
                    }) + "\n"
                    await asyncio.sleep(0.01)

                # Add to history
                history_store.add_interaction(session_id, query, answer)
                log.info(f"/rag Conversation history response completed for '{session_id}'")
                return
            
            # Use RAG chain for document-based queries
            rag_chain: Runnable = request.app.state.rag_chain
            history_store: HistoryStore = request.app.state.history_store
            retriever = request.app.state.retriever
            neo4j_retriever = getattr(request.app.state, 'neo4j_retriever', None)
            
            print(f"[DEBUG] Starting V2 retrieval for query: {query[:50]}")
            print("STEP 1: About to import planner")
            
            # Get timing breakpoints
            retrieval_start = time.time()
            
            # --- FILE-AWARE STRATEGY SELECTION ---
            # If user explicitly selected files, force strategy based on where those files are stored
            forced_strategy = None
            if selected_files:
                print(f"[DEBUG] Files explicitly selected: {selected_files}. Checking ingestion targets...")
                try:
                    file_targets = []
                    for filename in selected_files:
                        file_id = pg_db.get_file_id_by_name(user_id=session_id, file_name=filename)
                        if file_id != -1:
                            target = pg_db.get_ingestion_target(file_id)
                            file_targets.append(target)
                            print(f"[DEBUG] File '{filename}' ingestion_target: {target}")
                    
                    # Determine forced strategy based on file targets
                    if file_targets:
                        if all(t == 'neo4j' for t in file_targets):
                            forced_strategy = 'graph'
                            print(f"[DEBUG] All selected files are in Neo4j. Forcing strategy='graph'.")
                        elif all(t == 'qdrant' for t in file_targets):
                            forced_strategy = 'hybrid'
                            print(f"[DEBUG] All selected files are in Qdrant. Forcing strategy='hybrid'.")
                        else:  # Mix of targets or 'both'
                            forced_strategy = 'both'
                            print(f"[DEBUG] Mixed ingestion targets. Forcing strategy='both'.")
                except Exception as e:
                    print(f"[WARN] Failed to check ingestion targets: {e}. Will use Planner.")
            
            # --- PLANNER INTEGRATION (Only if no forced strategy) ---
            if forced_strategy:
                strategy = forced_strategy
                print(f"[DEBUG] Using forced strategy: {strategy} (file explicitly selected)")
            else:
                print(f"[DEBUG] No files selected or unable to determine target. Using Planner.")
                from src.agent.planner import create_retrieval_planner, get_document_summaries
                
                # 1. Fetch Summaries
                summaries_context = "No summaries available."
                try:
                    if retriever and retriever.qdrant_client:
                        summaries_context = get_document_summaries(
                            query=query,
                            qdrant_client=retriever.qdrant_client,
                            embeddings=retriever.embeddings
                        )
                except Exception as e:
                    print(f"[WARN] Failed to fetch summaries for planner: {e}")

                # 2. Plan Strategy
                try:
                    planner = create_retrieval_planner(llm_model=config.LLM_CHAT_MODEL_NAME)
                    plan = planner.invoke({
                        "query": query,
                        "summaries": summaries_context
                    })
                    strategy = plan.strategy
                    print(f"[DEBUG] Planner Strategy: {strategy} (Reason: {plan.reasoning})")
                except Exception as e:
                    print(f"[WARN] Planner failed, defaulting to hybrid: {e}")
                    strategy = "hybrid"

            # 3. Execute Retrieval
            retrieved_docs = []
            if strategy == "graph" and neo4j_retriever:
                print("[DEBUG] Executing Graph Retrieval")
                # Pass selected_files as kwarg (handled by custom get_relevant_documents or wrapper)
                retrieved_docs = neo4j_retriever.get_relevant_documents(query, selected_files=selected_files)
            elif strategy == "both" and neo4j_retriever:
                print("[DEBUG] Executing Hybrid + Graph Retrieval")
                hybrid_docs = retriever.retrieve_hybrid(query, selected_files)
                graph_docs = neo4j_retriever.get_relevant_documents(query, selected_files=selected_files)
                retrieved_docs = hybrid_docs + graph_docs
            else:
                print("[DEBUG] Executing Hybrid Retrieval")
                retrieved_docs = retriever.retrieve_hybrid(query, selected_files)

            retrieval_time = round(time.time() - retrieval_start, 2)
            
            print(f"[DEBUG] Retrieved {len(retrieved_docs)} documents in {retrieval_time}s")
            
            # Send retrieved chunks immediately
            chunks_data = []
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"[DEBUG] Processing doc #{i}: metadata={doc.metadata}")
                
                # Filter out system initialization docs
                if doc.metadata.get('skip_in_search'):
                    print(f"[DEBUG] Skipping doc #{i}: has skip_in_search flag")
                    continue
                    
                # Clean up content: strip whitespace, normalize spaces, collapse newlines
                content = doc.page_content[:500]
                content = ' '.join(content.split())  # Normalize whitespace
                if len(doc.page_content) > 500:
                    content += "..."
                
                # Get source document name
                source = doc.metadata.get('source', doc.metadata.get('filename', 'unknown'))
                
                print(f"[DEBUG] Doc #{i}: source='{source}', content_length={len(content)}, first_50_chars='{content[:50]}'")
                
                # Filter out "unknown" sources if they are empty/garbage
                if source == 'unknown' and len(content) < 5:
                    print(f"[DEBUG] Skipping doc #{i}: unknown source with short content (len={len(content)})")
                    continue
                
                print(f"[DEBUG] Chunk #{i}: source={source}, content_length={len(content)}, raw_length={len(doc.page_content)} - KEEPING THIS ONE")
                
                chunks_data.append({
                    "rank": i,
                    "filename": doc.metadata.get('filename', 'unknown'),
                    "source": source,
                    "content": content if content else "(empty content)",
                    "metadata": {
                        **{k: v for k, v in doc.metadata.items() if k not in ['embedding', 'source', 'filename']},
                        "score": doc.metadata.get('graph_score') or doc.metadata.get('rrf_score') or doc.metadata.get('semantic_score') or 0.0
                    }
                })
            
            print(f"[DEBUG] Prepared {len(chunks_data)} chunks for sending")
            
            yield json.dumps({
                "type": "chunks",
                "data": {
                    "retrieved": chunks_data,
                    "retrieval_time": retrieval_time
                }
            }) + "\n"
            
            print(f"[DEBUG] Sent chunks data to client")
            
            # Configure with thread_id for LangGraph checkpointer
            # The checkpointer automatically handles loading/saving messages
            # checkpoint_ns is required for AsyncRedisSaver to function correctly!
            run_config = {
                "configurable": {
                    "thread_id": session_id,  # LangGraph uses thread_id for memory
                    "checkpoint_ns": "",      # Required by AsyncRedisSaver
                    "selected_files": selected_files
                }
            }
            
            # Stream the response
            llm_start = time.time()
            full_response = ""
            chunk_count = 0
            
            initial_state = {
                "input": query,
                "selected_files": selected_files,
                "messages": [HumanMessage(content=query)]
            }
            
            log.info(f"RAG Endpoint: Starting astream for query '{query[:50]}...' session '{session_id}'")
            try:
                async for chunk in rag_chain.astream(initial_state, config=run_config):
                    chunk_count += 1
                    # log.info(f"RAG Endpoint: Received chunk #{chunk_count}: {list(chunk.keys()) if isinstance(chunk, dict) else type(chunk)}")
                    
                    if await request.is_disconnected():
                        log.warning(f"/rag client disconnected for '{session_id}'")
                        break
                        
                    # Extract answer from chunk (LangGraph yields node outputs)
                    # Expected format: {'generator': {'answer': '...'}} OR just {'answer': '...'} properties
                    
                    content = None
                    if isinstance(chunk, dict):
                        # check for nested generator output
                        if "generator" in chunk and isinstance(chunk["generator"], dict):
                             content = chunk["generator"].get("answer")
                        elif "answer" in chunk:
                             content = chunk["answer"]
                             
                    if content:
                        # print(f"[DEBUG] Yielding content: {content[:50]}...")
                        full_response += content
                        yield json.dumps({
                            "type": "content",
                            "data": content
                        }) + "\n"
                        # print(f"[DEBUG] Yielded LLM chunk #{chunk_count} to client")
            except Exception as stream_error:
                print(f"[ERROR] LLM streaming error: {stream_error}")
                log.error(f"/rag LLM streaming error: {stream_error}", exc_info=True)
                yield json.dumps({
                    "type": "error",
                    "data": {"message": f"LLM error: {str(stream_error)}"}
                }) + "\n"
            
            print(f"[DEBUG] LLM streaming completed. Chunks received: {chunk_count}")
            llm_time = round(time.time() - llm_start, 2)
            
            # Send sources if available
            if selected_files:
                yield json.dumps({
                    "type": "sources",
                    "data": {"files": selected_files}
                }) + "\n"
            
            # Add to history
            history_store.add_interaction(session_id, query, full_response)
            
            # Send duration metadata with breakdown
            total_duration = round(time.time() - start_time, 2)
            yield json.dumps({
                "type": "duration",
                "data": {
                    "seconds": total_duration,
                    "breakdown": {
                        "retrieval": retrieval_time,
                        "llm": llm_time,
                        "other": round(total_duration - retrieval_time - llm_time, 2)
                    }
                }
            }) + "\n"
            
            log.info(f"/rag Streaming completed for '{session_id}' in {total_duration}s (retrieval={retrieval_time}s, llm={llm_time}s)")
            
        except Exception as e:
            log.exception(f"/rag Error {e} for '{session_id}'")
            yield json.dumps({
                "type": "error",
                "data": str(e)
            }) + "\n"
    
    return StreamingResponse(rag_streamer(), media_type="text/plain")


# ------------------------------------------------------------------------------
# Initialization End-points:
# ------------------------------------------------------------------------------

# Helper function to delete old files and embeddings:
def delete_old_files(user_id: str, time: int = OLD_FILE_THRESHOLD):
    """Function to delete old files and embeddings older than the specified time."""
    import sys
    print(f"/delete DELETE CALLED for user '{user_id}'", file=sys.stderr, flush=True)
    log.info(
        f"/delete Deleting old files and embeddings for user '{user_id}' older than {time} seconds")

    # Delete old files
    old_files = pg_db.get_old_files(user_id=user_id, time=time)
    print(f"/delete OLD_FILES RESULT: {old_files}", file=sys.stderr, flush=True)
    log.info(f"/delete Got old_files result: {old_files}")
    if old_files['files']:
        log.info(f"/delete Removing old files for user '{user_id}': {old_files['files']}")

        for file in old_files['files']:
            # Try to delete physical file
            file_deleted = files.delete_file(user_id=user_id, file_name=file)
            log.info(f"/delete Deleted physical file '{file}': {file_deleted}")
            
            # Always mark file as removed in database (whether physical deletion succeeded or not)
            file_id = pg_db.get_file_id_by_name(user_id=user_id, file_name=file)
            log.info(f"/delete Got file_id {file_id} for file '{file}'")
            db_marked = pg_db.mark_file_removed(user_id=user_id, file_id=file_id)
            log.info(f"/delete Marked file '{file}' (ID {file_id}) as removed in DB: {db_marked}")
            
            if not file_deleted:
                log.warning(f"/delete Physical file '{file}' could not be deleted, but DB record marked as removed")

    # Delete old embeddings
    if old_files['embeddings']:
        log.info(f"/delete Removing {len(old_files['embeddings'])} old embeddings for user '{user_id}': {old_files['embeddings'][:5]}")
        vs: VectorDB = app.state.vector_db
        db: T_VECTOR_STORE = vs.get_vector_store()
        
        # Try to delete from Qdrant vector store (optional - vector store deletion can fail without breaking the system)
        qdrant_success = False
        try:
            log.info(f"/delete Attempting to delete {len(old_files['embeddings'])} embeddings from Qdrant...")
            resp = db.delete(old_files['embeddings'])
            log.info(f"/delete Qdrant deletion response: {resp}")
            qdrant_success = True
        except Exception as e:
            log.warning(f"/delete Failed to delete embeddings from Qdrant: {type(e).__name__}: {e}. Will proceed with database update.")
            qdrant_success = False
        
        # ALWAYS mark embeddings as removed in PostgreSQL database for data consistency
        # This ensures the embeddings table reflects reality even if Qdrant deletion had issues
        log.info(f"/delete Marking {len(old_files['embeddings'])} embeddings as unavailable in PostgreSQL...")
        db_marked = pg_db.mark_embeddings_removed(vector_ids=old_files['embeddings'])
        log.info(f"/delete PostgreSQL mark_embeddings_removed result: {db_marked}")
        
        if db_marked:
            log.info(f"/delete ✓ Successfully marked {len(old_files['embeddings'])} embeddings as unavailable in PostgreSQL")
            if qdrant_success:
                log.info(f"/delete ✓ Also successfully removed from Qdrant vector store")
            else:
                log.warning(f"/delete ⚠ Removed from PostgreSQL but Qdrant deletion failed (OK - data is marked unavailable)")
        else:
            log.error(f"/delete ✗ Failed to mark embeddings as unavailable in PostgreSQL")
    else:
        log.info(f"/delete No old embeddings found for user '{user_id}'")


# First end-point to call on client initialization:
class LoginRequest(BaseModel):
    login_id: str
    password: str


@app.post("/login")
async def login(request: Request, login_request: LoginRequest):
    """Endpoint to handle user login.
    + Client sends login_id and password for login
    + Based on it, server authenticates user.
    + user_id is retrieved (for now, it is same as login_id)
    + ? Based on user_id chat history of user is retrieved n returned.

    * Folder is created for user_id, older files are removed
    * Later on, will add on scheduled job to delete old items and will remove the old file deletion logic from here.

    - Post request expects JSON `{"login_id": "", "password": ""}` structure.
    - Return JSON with `{"user_id": "user_id", "chat_history": [user chat history]}` structure.
    """

    login_id = login_request.login_id.strip()
    password = login_request.password.strip()
    log.info(f"/login Requested by '{login_id}'")

    # Check if the user exists in the database
    status, msg = pg_db.authenticate_user(user_id=login_id, password=password)
    if status:
        user_id = login_id
        # Check if folder exists in UPLOADS_DIR with user_id
        files.create_user_uploads_folder(user_id=user_id)
        # Skip old file deletion to avoid Qdrant format errors
        # delete_old_files(user_id=user_id, time=OLD_FILE_THRESHOLD)
        return JSONResponse(content={"user_id": user_id, "name": msg}, status_code=200)
    else:
        return JSONResponse(content={"error": msg}, status_code=401)

    # # For now, we will just return a dummy user_id
    # # In future, can implement actual user authentication and return a real user_id
    # user_id = login_id
    # log.info(f"/login requested by '{user_id}'")

    # # Check if folder exists in UPLOADS_DIR with user_id
    # files.create_user_uploads_folder(user_id=user_id)

    # # Old any older data if exists (older than 24 hours)
    # delete_old_files(user_id=user_id, time=OLD_FILE_THRESHOLD)

    # # Get the chat history for the user_id
    # hs: HistoryStore = request.app.state.history_store
    # history = hs.get_session_history(session_id=user_id)
    # if not history:
    #     log.info(f"/login No history found for user '{user_id}'")
    # else:
    #     log.info(f"/login History found for user '{user_id}' with {len(history.messages)} messages")

    # return {"user_id": user_id, "chat_history": history.messages}


# endpoint for user registration:
class RegisterRequest(BaseModel):
    name: str
    user_id: str
    password: str


@app.post("/register")
async def register(request: Request, register_request: RegisterRequest):
    """Endpoint to handle user registration.
    - Post request expects JSON `{"user_name": "Full Name", "user_id": "any_u_id", "password": "raw_pw"}` structure.
    - Return JSON with `{"status": "success"}` or `{"error": "message"}` structure.
    """

    name = register_request.name.strip()
    user_id = register_request.user_id.strip()
    password = register_request.password.strip()
    log.info(f"/register Requested by {name} with '{user_id}'")
    print(f"Name: {name}, UserID: {user_id}, Password: {password}")

    # Check if the user already exists
    status = pg_db.check_user_exists(user_id=user_id)
    if status:
        log.error(f"/register UserID '{user_id}' already exists.")
        return JSONResponse(content={"error": "User already exists"}, status_code=400)

    # If user does not exist, add the user to the database
    status = pg_db.add_user(user_id=user_id, name=name, password=password)
    if status:
        return JSONResponse(content={"status": "success"}, status_code=201)
    else:
        return JSONResponse(content={"error": "Failed to register user"}, status_code=500)


# ------------------------------------------------------------------------------
# Chat History Endpoints:
# ------------------------------------------------------------------------------

# Endpoint to get chat history for user:
@app.post("/chat_history")
async def chat_history(user_id: str = Form(...)):
    """Endpoint to get chat history for user.
    - Post request expects `user_id` as form parameter.
    - Return JSON with `{"chat_history": [user chat history]}` or `{"error": "message"}` structure.
    """
    log.info(f"/chat_history Requested by '{user_id}'")
    hs: HistoryStore = app.state.history_store
    history = hs.get_session_history(session_id=user_id)

    if history:
        messages = []
        for msg in history.messages:
            msg: T_MESSAGE
            if msg.type == "ai":
                messages.append({"role": "assistant", "content": msg.text()})
            elif msg.type == "human":
                messages.append({"role": "human", "content": msg.text()})

        return JSONResponse(content={"chat_history": messages}, status_code=200)
    else:
        return JSONResponse(content={"error": "No chat history found"}, status_code=404)


# Endpoint /clear_chat_history to clear chat history for user:
@app.post("/clear_chat_history")
async def clear_chat_history(user_id: str = Form(...)):
    """Endpoint to clear chat history for user.
    - Post request expects `user_id` as form parameter.
    - Return JSON with `{"status": "success"}` or `{"error": "message"}` structure.
    """
    log.info(f"/clear_chat_history Requested by '{user_id}'")
    hs: HistoryStore = app.state.history_store
    status = hs.clear_session_history(session_id=user_id)

    if status:
        return JSONResponse(content={"status": "success"}, status_code=200)
    else:
        return JSONResponse(content={"error": "No history found to clear"}, status_code=404)


# ------------------------------------------------------------------------------
# File handling endpoints:
# ------------------------------------------------------------------------------

# Endpoint to check if file needs embedding (resume capability):
@app.post("/check_file_status")
async def check_file_status(user_id: str = Form(...), filename: str = Form(...)):
    """Endpoint to check if a file needs embedding.
    - Returns: {'status': 'exists' | 'needs_embedding' | 'complete' | 'not_found'}
    - 'exists': File in uploads table but needs embedding
    - 'needs_embedding': Same file detected, ready to embed
    - 'complete': File has embeddings
    - 'not_found': File doesn't exist
    """
    log.info(f"/check_file_status Checking file '{filename}' for user '{user_id}'")
    
    try:
        # Get file ID if it exists
        file_id = pg_db.get_file_id_by_name(user_id=user_id, file_name=filename)
        
        if file_id is None:
            return JSONResponse(
                content={"status": "not_found", "message": "File not found"},
                status_code=200
            )
        
        # Check if file has embeddings
        embeddings = pg_db.get_embeddings_by_file_id(file_id)
        
        if embeddings and len(embeddings) > 0:
            return JSONResponse(
                content={
                    "status": "complete",
                    "message": "File already has embeddings",
                    "file_id": file_id,
                    "embedding_count": len(embeddings)
                },
                status_code=200
            )
        else:
            return JSONResponse(
                content={
                    "status": "needs_embedding",
                    "message": "File exists but needs embedding",
                    "file_id": file_id
                },
                status_code=200
            )
    except Exception as e:
        log.error(f"/check_file_status Error checking file status: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

# Endpoint to receive file uploads:
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    user_id: str = Form(...),
    use_graph: str = Form("false") # Received as string from FormData
):
    log.info(f"/upload Received file: {file.filename} from user: {user_id}")
    filename = file.filename if file.filename else "unknown_file"
    
    # Convert use_graph string to boolean
    use_graph_bool = use_graph.lower() == "true"
    if use_graph_bool:
        log.info(f"/upload Graph ingestion enabled for {filename}")
    
    try:
        # Ensure user exists in the database
        if not pg_db.check_user_exists(user_id):
            # Create user automatically if not exists
            pg_db.add_user(user_id=user_id, name=user_id, password="")
            log.info(f"/upload Created new user: {user_id}")

        # Read file content to get size and type
        file_content = await file.read()
        file_size = len(file_content)
        file_type = file.content_type or "application/octet-stream"
        
        # Validate file size
        max_size_mb = 500
        if file_size > max_size_mb * 1024 * 1024:
            error_msg = f"File too large. Maximum size is {max_size_mb}MB"
            log.warning(f"/upload {error_msg} for user {user_id}")
            return JSONResponse(status_code=400, content={"error": error_msg})

        # Save the file to disk
        success, saved_filename = files.save_file(user_id, file_content, filename)
        
        if not success:
            error_msg = saved_filename  # save_file returns error message in second tuple element on failure
            log.error(f"/upload Failed to save file: {error_msg}")
            return JSONResponse(status_code=500, content={"error": f"Failed to save file: {error_msg}"})
            
        file_path = files.get_file_path(user_id, saved_filename)
        log.info(f"/upload File saved successfully at {file_path}")
        
        # Extract metadata (creation time, modification time)
        try:
            # For uploaded files, we use current time as creation/mod time since we don't get original fs metadata
            import datetime
            created_at = datetime.datetime.now()
            modified_at =  datetime.datetime.now()
            
            # If we wanted to be more precise we could try to get it from os.stat but it's just created
            # stat = os.stat(file_path)
            # created_at = datetime.datetime.fromtimestamp(stat.st_ctime)
            # modified_at = datetime.datetime.fromtimestamp(stat.st_mtime)
            
        except Exception as meta_err:
            log.warning(f"/upload Failed to extract metadata: {str(meta_err)}, using current time")
            import datetime
            created_at = datetime.datetime.now()
            modified_at = datetime.datetime.now()

        # Add file entry to database
        try:
            file_id = pg_db.add_file(
                user_id=user_id, 
                filename=saved_filename, 
                file_size=file_size,
                file_type=file_type
            )
            log.info(f"/upload Added file entry to DB: ID={file_id}")
        except Exception as db_err:
            # Check for duplicate
            if "unique constraint" in str(db_err).lower() or "already exists" in str(db_err).lower():
                log.info(f"/upload File entry already exists in DB, retrieving ID...")
                file_id = pg_db.get_file_id_by_name(user_id, saved_filename)
                if file_id == -1:
                    raise db_err
            else:
                raise db_err

        # Trigger background IDEMPOTENT ingestion task
        try:
            task = ingest_file_task.delay(
                user_id=user_id,
                file_name=saved_filename,
                collection_name=config.QDRANT_COLLECTION_NAME,
                embedding_model=config.EMB_MODEL_NAME,
                use_graph=use_graph_bool
            )
            log.info(f"/upload Auto-triggered embedding task {task.id} for {filename} (Graph={use_graph_bool})")
            
            return {
                "message": "File uploaded successfully", 
                "filename": saved_filename,
                "file_id": file_id,
                "task_id": str(task.id),
                "status": "uploaded",
                "embedding_status": "embedding_queued"
            }
            
        except Exception as embed_err:
            log.error(f"/upload Failed to queue embedding task: {str(embed_err)}", exc_info=True)
            return JSONResponse(
                status_code=202, 
                content={
                    "message": "File uploaded but auto-ingestion failed to start. Please try manually.",
                    "filename": saved_filename,
                    "file_id": file_id,
                    "error": str(embed_err)
                }
            )

    except Exception as e:
        error_msg = f"Unexpected error during upload"
        log.error(f"/upload {error_msg} | Original error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"{error_msg}: {str(e)}"})


@app.get("/uploads")
async def get_uploads(user_id: str = Query(...)):
    """Get all uploaded files for a user."""
    try:
        uploads_data = pg_db.get_uploads(user_id=user_id)
        return JSONResponse(content=uploads_data, status_code=200)
    except Exception as e:
        log.error(f"/uploads Error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to fetch uploads: {str(e)}"},
            status_code=500
        )


@app.delete("/delete_all_files")
async def delete_all_files_endpoint(user_id: str = Query(...)):
    """Delete all files and embeddings for a user."""
    log.info(f"/delete_all_files Deleting all files for user '{user_id}'")
    
    try:
        # Get all files for the user
        user_files = pg_db.get_files_by_user(user_id)
        
        if not user_files:
            log.info(f"/delete_all_files No files found for user '{user_id}'")
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "No files to delete",
                    "files_deleted": 0,
                    "embeddings_deleted": 0
                },
                status_code=200
            )
        
        total_files = len(user_files)
        total_embeddings = 0
        failed_files = []
        
        vs: VectorDB = app.state.vector_db
        
        for file_record in user_files:
            file_id = file_record['id']
            filename = file_record['filename']
            
            try:
                # Get embedding IDs for this file
                embedding_ids = pg_db.get_embeddings_by_file_id(file_id)
                total_embeddings += len(embedding_ids)
                
                # Delete from Qdrant
                if embedding_ids:
                    try:
                        vs.delete_documents(embedding_ids)
                        pg_db.mark_embeddings_removed(embedding_ids)
                    except Exception as ve:
                        log.error(f"/delete_all_files Error deleting vectors for '{filename}': {ve}")
                
                # Delete physical file
                files.delete_file(user_id=user_id, file_name=filename)
                
                # Mark file as removed in database
                pg_db.mark_file_removed(user_id=user_id, file_id=file_id)
                
            except Exception as e:
                log.error(f"/delete_all_files Error deleting '{filename}': {e}")
                failed_files.append(filename)
        
        log.info(f"/delete_all_files Deleted {total_files - len(failed_files)}/{total_files} files for user '{user_id}'")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Deleted {total_files - len(failed_files)} files",
                "files_deleted": total_files - len(failed_files),
                "embeddings_deleted": total_embeddings,
                "failed_files": failed_files
            },
            status_code=200
        )
        
    except Exception as e:
        log.error(f"/delete_all_files Error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to delete files: {str(e)}"},
            status_code=500
        )


@app.delete("/delete_file")
async def delete_file_endpoint(user_id: str = Query(...), filename: str = Query(...)):
    """Delete a file and its embeddings for a user."""
    log.info(f"/delete_file Deleting file '{filename}' for user '{user_id}'")
    
    try:
        # Get file ID to retrieve embeddings before deletion
        file_id = pg_db.get_file_id_by_name(user_id=user_id, file_name=filename)
        
        if file_id == -1:
            log.warning(f"/delete_file File '{filename}' not found for user '{user_id}'")
            return JSONResponse(
                content={"error": "File not found"},
                status_code=404
            )
        
        # Get file metadata to determine ingestion target
        with pg_db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT ingestion_target FROM uploads WHERE file_id = %s",
                (file_id,)
            )
            result = cur.fetchone()
            ingestion_target = result[0] if result else 'qdrant'  # Default to qdrant if not set
        
        log.info(f"/delete_file File '{filename}' was ingested to: {ingestion_target}")
        
        # Get embedding IDs associated with this file
        embedding_ids = pg_db.get_embeddings_by_file_id(file_id)
        log.info(f"/delete_file Found {len(embedding_ids)} embeddings for file '{filename}'")
        
        # Delete from appropriate vector database
        if ingestion_target == 'neo4j':
            # Delete from Neo4j Graph Database
            try:
                from langchain_community.graphs import Neo4jGraph
                from config import settings as config
                
                graph = Neo4jGraph(
                    url=config.NEO4J_URI,
                    username=config.NEO4J_USERNAME,
                    password=config.NEO4J_PASSWORD
                )
                
                # Delete Document nodes and their relationships for this file
                # This includes chunks that were ingested from this file
                file_path = files.get_file_path(user_id=user_id, file_name=filename)
                
                # Delete all nodes that came from this source document
                # LangChain stores source info in Document nodes
                query = """
                MATCH (d:Document)
                WHERE d.source CONTAINS $filename OR d.text CONTAINS $filename
                DETACH DELETE d
                """
                result = graph.query(query, params={"filename": filename})
                log.info(f"/delete_file Deleted Neo4j graph nodes for file '{filename}'")
                
                # Also delete any entities extracted from this document
                # (They have MENTIONS relationships to Document nodes)
                orphan_cleanup = """
                MATCH (n)
                WHERE NOT (n)--()
                DELETE n
                """
                graph.query(orphan_cleanup)
                log.info(f"/delete_file Cleaned up orphaned Neo4j nodes")
                
            except Exception as neo4j_error:
                log.error(f"/delete_file Error deleting from Neo4j: {neo4j_error}")
                # Continue with other cleanup even if Neo4j fails
        else:
            # Delete from vector database (Qdrant)
            if embedding_ids:
                vs: VectorDB = app.state.vector_db
                try:
                    deleted_count = vs.delete_documents(embedding_ids)
                    log.info(f"/delete_file Deleted {deleted_count} vectors from Qdrant for file '{filename}'")
                except Exception as ve:
                    log.error(f"/delete_file Error deleting vectors from Qdrant: {ve}")
                    # Continue with database cleanup even if vector deletion fails
        
        # Delete document summary from Qdrant (applies to both ingestion targets)
        try:
            file_path = files.get_file_path(user_id=user_id, file_name=filename)
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            vs: VectorDB = app.state.vector_db
            # Delete summary by filtering on file_path
            vs.db.client.delete(
                collection_name="document_summaries",
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path)
                        )
                    ]
                )
            )
            log.info(f"/delete_file Deleted document summary from Qdrant for file '{filename}'")
        except Exception as summary_error:
            log.warning(f"/delete_file Error deleting summary: {summary_error}")
        
        # Mark embeddings as removed in database
        if embedding_ids:
            pg_db.mark_embeddings_removed(embedding_ids)
        
        # Delete physical file from disk
        file_deleted = files.delete_file(user_id=user_id, file_name=filename)
        if not file_deleted:
            log.warning(f"/delete_file Physical file '{filename}' could not be deleted from disk")
        
        # Mark file as removed in database
        db_marked = pg_db.mark_file_removed(user_id=user_id, file_id=file_id)
        
        if db_marked:
            log.info(f"/delete_file Successfully deleted file '{filename}' for user '{user_id}'")
            return JSONResponse(
                content={
                    "status": "success",
                    "message": f"File '{filename}' deleted successfully from {ingestion_target}",
                    "embeddings_deleted": len(embedding_ids)
                },
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": "Failed to mark file as deleted in database"},
                status_code=500
            )
            
    except Exception as e:
        log.error(f"/delete_file Error deleting file '{filename}': {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to delete file: {str(e)}"},
            status_code=500
        )


# Endpoint to embed the uploaded file:
# takes user_id and file_name as input
class EmbedRequest(BaseModel):
    user_id: str
    file_name: str
    embedding_model: Optional[str] = None


class ReembedAllRequest(BaseModel):
    user_id: str
    embedding_model: Optional[str] = None


@app.post("/reembed-all")
async def reembed_all_files(req: ReembedAllRequest):
    """Re-embed all files for a user with the specified embedding model."""
    user_id = req.user_id.strip()
    embedding_model = req.embedding_model or config.EMB_MODEL_NAME
    
    log.info(f"/reembed-all Requested by '{user_id}' using model '{embedding_model}'")
    
    try:
        # Get all files for the user
        uploads_data = pg_db.get_uploads(user_id=user_id)
        files_list = uploads_data.get('files', [])
        
        if not files_list:
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "No files found to re-embed",
                    "count": 0
                },
                status_code=200
            )
        
        # Clear all embeddings for this user
        log.info(f"Marking all files for user '{user_id}' for re-embedding")
        for file_dict in files_list:
            file_name = file_dict.get('filename')
            file_id = pg_db.get_file_id_by_name(user_id=user_id, file_name=file_name)
            if file_id != -1:
                pg_db.clear_embeddings_for_file(file_id)
                pg_db.set_embedding_status(file_id, 'pending')
                pg_db.update_embedding_model(user_id=user_id, filename=file_name, embedding_model=embedding_model)
        
        # Trigger embedding tasks for all files
        task_ids = []
        for file_dict in files_list:
            file_name = file_dict.get('filename')
            try:
                task = ingest_file_task.delay(
                    user_id=user_id,
                    file_name=file_name,
                    collection_name="documents",
                    embedding_model=embedding_model
                )
                task_ids.append({"filename": file_name, "task_id": task.id})
                log.info(f"Triggered re-embedding task {task.id} for {file_name}")
            except Exception as e:
                log.error(f"Failed to trigger re-embedding for {file_name}: {e}")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Re-embedding triggered for {len(task_ids)} files",
                "count": len(task_ids),
                "tasks": task_ids
            },
            status_code=202
        )
        
    except Exception as e:
        log.error(f"/reembed-all Error: {e}", exc_info=True)
        return JSONResponse(
            content={
                "error": f"Failed to trigger re-embedding: {str(e)}"
            },
            status_code=500
        )


@app.post("/embed")
async def embed_file(embed_request: EmbedRequest, request: Request):
    """Endpoint to embed the uploaded file (async/background task).
    - Post request expects JSON `{"user_id": "", "file_name": "", "embedding_model": "optional"}` structure.
    - Return JSON with `{"status": "accepted", "task_id": "...", "message": "..."}` structure.
    - Use /embed/status/{task_id} endpoint to check progress.
    """
    user_id = embed_request.user_id.strip()
    file_name = embed_request.file_name.strip()
    embedding_model = embed_request.embedding_model or config.EMB_MODEL_NAME

    # Update the embedding model in the database
    pg_db.update_embedding_model(user_id=user_id, filename=file_name, embedding_model=embedding_model)

    log.info(f"/embed Requested by '{user_id}' for file '{file_name}' using model '{embedding_model}'")

    try:
        # Submit async task to Celery (pass file_name, worker will construct path)
        task = ingest_file_task.delay(
            user_id=user_id,
            file_name=file_name,
            collection_name="documents",
            embedding_model=embedding_model
        )
        
        log.info(f"/embed Background task '{task.id}' submitted for '{user_id}' and file '{file_name}'")
        
        return JSONResponse(
            content={
                "status": "accepted",
                "task_id": task.id,
                "message": f"File ingestion started in background. Use /embed/status/{task.id} to check progress.",
                "user_id": user_id,
                "file_name": file_name
            },
            status_code=202  # Accepted
        )
    except FileNotFoundError as e:
        error_msg = f"File '{file_name}' not found for user '{user_id}'. Please upload the file first."
        log.error(f"/embed {error_msg}", exc_info=True)
        return JSONResponse(
            content={
                "error": error_msg,
                "error_type": "file_not_found",
                "details": str(e)
            },
            status_code=404
        )
    except Exception as e:
        error_str = str(e).lower()
        error_type = "unknown_error"
        error_msg = f"Failed to start ingestion: {str(e)}"
        
        # Provide specific error messages based on exception type
        if "redis" in error_str or "connection" in error_str:
            error_type = "redis_unavailable"
            error_msg = "Background task queue (Redis) is not available. Please ensure the task broker is running."
        elif "celery" in error_str:
            error_type = "celery_unavailable"
            error_msg = "Celery worker is not available. Please ensure the worker is running."
        elif "database" in error_str or "postgresql" in error_str:
            error_type = "database_error"
            error_msg = "Database connection failed. Please check PostgreSQL is running."
        elif "qdrant" in error_str:
            error_type = "qdrant_unavailable"
            error_msg = "Vector database (Qdrant) is not available. Please ensure Qdrant is running."
        elif "timeout" in error_str:
            error_type = "timeout_error"
            error_msg = "Request timeout. The system is under heavy load. Please try again."
        
        log.error(f"/embed {error_msg} | Original error: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "error": error_msg,
                "error_type": error_type,
                "details": str(e)
            },
            status_code=500
        )


# New endpoint to check embedding task status:
@app.get("/embed/status/{task_id}")
async def embed_status(task_id: str, request: Request):
    """Endpoint to check the status of an embedding task.
    - Get request with task_id from path.
    - Return JSON with task status and progress information.
    """
    from celery_app import app as celery_app
    
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            status_response = {
                'task_id': task_id,
                'status': 'pending',
                'progress': 0,
                'message': 'Task is waiting to be processed...'
            }
        elif result.state == 'PROGRESS':
            status_response = {
                'task_id': task_id,
                'status': 'processing',
                'progress': result.info.get('current', 0),
                'total': result.info.get('total', 100),
                'message': result.info.get('status', 'Processing...')
            }
        elif result.state == 'SUCCESS':
            status_response = {
                'task_id': task_id,
                'status': 'completed',
                'progress': 100,
                'message': 'Ingestion completed successfully!',
                'doc_count': len(result.result.get('doc_ids', [])) if isinstance(result.result, dict) else 0,
                'result': result.result
            }
        elif result.state == 'FAILURE':
            # Extract detailed error information
            error_msg = str(result.info)
            error_type = "task_failed"
            
            # Parse specific error types from traceback
            if "redis" in error_msg.lower():
                error_type = "redis_error"
                user_msg = "Failed to queue task. Redis broker is not responding. Check if Redis is running."
            elif "qdrant" in error_msg.lower():
                error_type = "qdrant_error"
                user_msg = "Failed to store embeddings. Vector database (Qdrant) is not responding."
            elif "ollama" in error_msg.lower():
                error_type = "ollama_error"
                user_msg = "Failed to generate embeddings. LLM service (Ollama) is not responding."
            elif "database" in error_msg.lower() or "postgresql" in error_msg.lower():
                error_type = "database_error"
                user_msg = "Database operation failed. Check PostgreSQL connection."
            elif "timeout" in error_msg.lower():
                error_type = "timeout_error"
                user_msg = "Task timed out. The file may be too large or system is under heavy load."
            elif "out of memory" in error_msg.lower():
                error_type = "out_of_memory"
                user_msg = "Insufficient memory to process file. Try with a smaller file."
            else:
                user_msg = f"Task failed: {error_msg[:200]}"
            
            status_response = {
                'task_id': task_id,
                'status': 'failed',
                'progress': 0,
                'message': user_msg,
                'error_type': error_type,
                'error_details': error_msg,
                'retry_suggested': True
            }
        else:
            status_response = {
                'task_id': task_id,
                'status': result.state.lower(),
                'progress': 0,
                'message': f'Task state: {result.state}'
            }
        
        log.info(f"/embed/status Task '{task_id}' status: {status_response['status']}")
        return JSONResponse(content=status_response, status_code=200)
        
    except Exception as e:
        log.error(f"/embed/status Error checking status for task '{task_id}': {str(e)}")
        return JSONResponse(
            content={"error": f"Failed to check task status: {str(e)}"},
            status_code=500
        )


# ------------------------------------------------------------------------------
# Model Management Endpoints:
# ------------------------------------------------------------------------------

class ModelRequest(BaseModel):
    name: str

class ModelSelectRequest(BaseModel):
    type: str # 'llm' or 'embedding'
    name: str

@app.post("/models/pull")
async def pull_model(request: ModelRequest):
    """Trigger Ollama to pull a model.
    Note: This returns immediately, but the pull happens on the Ollama server.
    Currently, we don't stream the progress back, but in a production app you would.
    """
    model_name = request.name.strip()
    log.info(f"/models/pull Requesting pull for '{model_name}'")
    
    try:
        url = f"{config.OLLAMA_BASE_URL}/api/pull"
        data = json.dumps({"name": model_name}).encode()
        
        # We'll use a fire-and-forget or a simple initial request check
        # For simplicity in this demo, we assume the user will wait or check back
        req = urllib.request.Request(url, data=data, method="POST")
        
        with urllib.request.urlopen(req, timeout=10) as response:
            # We just read the first chunk to ensure it started
            # Ollama streams the response...
             return {"status": "started", "message": f"Pull started for {model_name}"}

    except Exception as e:
        log.error(f"/models/pull Error: {e}")
        return JSONResponse(
            content={"error": f"Failed to start pull: {str(e)}"},
            status_code=500
        )

@app.delete("/models/delete")
async def delete_model(name: str):
    """Trigger Ollama to delete a model."""
    model_name = name.strip()
    log.info(f"/models/delete Requesting delete for '{model_name}'")
    
    try:
        url = f"{config.OLLAMA_BASE_URL}/api/delete"
        data = json.dumps({"name": model_name}).encode()
        
        req = urllib.request.Request(url, data=data, method="DELETE")
        
        with urllib.request.urlopen(req, timeout=10) as response:
             if response.status == 200:
                 return {"status": "success", "message": f"Model {model_name} deleted"}
             else:
                 return JSONResponse(content={"error": "Ollama delete failed"}, status_code=response.status)

    except Exception as e:
         log.error(f"/models/delete Error: {e}")
         return JSONResponse(
            content={"error": f"Failed to delete model: {str(e)}"},
            status_code=500
        )

# Get active models from config
def get_active_models():
    return {
        "llm": config.LLM_CHAT_MODEL_NAME,
        "embedding": config.EMB_MODEL_NAME
    }

active_models = get_active_models()

@app.post("/models/select")
async def select_model(req: ModelSelectRequest):
    """Set the active model for LLM or Embedding."""
    if req.type not in ["llm", "embedding"]:
        return JSONResponse(content={"error": "Invalid type"}, status_code=400)
    
    old_model = active_models.get(req.type)
    active_models[req.type] = req.name
    log.info(f"/models/select Set active {req.type} model to '{req.name}' (was: '{old_model}')")
    
    # If it's embedding, we might need to update global config or re-init components
    # For now, we update the config module's variable (runtime patch)
    if req.type == "embedding":
        model_changed = old_model and old_model != req.name
        config.EMB_MODEL_NAME = req.name
        
        # Smart logic: If embedding model changed, clear embeddings and trigger re-embedding
        if model_changed:
            log.warning(f"Embedding model changed from '{old_model}' to '{req.name}'")
            log.info("Clearing all embeddings and marking files for re-embedding...")
            
            try:
                # Clear all embeddings from database
                from src.core.database import clear_all_embeddings, mark_all_files_for_reembedding
                cleared_count = clear_all_embeddings()
                log.info(f"Cleared {cleared_count} embedding records from database")
                
                # Mark all files as pending for re-embedding
                marked_count = mark_all_files_for_reembedding()
                log.info(f"Marked {marked_count} files for re-embedding")
                
                # Delete Qdrant collection to start fresh with new dimensions
                try:
                    import requests
                    qdrant_url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
                    response = requests.delete(f"{qdrant_url}/collections/documents", timeout=5)
                    if response.status_code == 200:
                        log.info("Deleted Qdrant collection 'documents'")
                    else:
                        log.warning(f"Could not delete Qdrant collection: {response.text}")
                except Exception as e:
                    log.warning(f"Could not delete Qdrant collection: {e}")
                
                return {
                    "status": "success", 
                    "active_models": active_models,
                    "message": f"Embedding model changed. Cleared {cleared_count} embeddings. Please re-upload or re-embed {marked_count} files.",
                    "requires_reembedding": True
                }
            except Exception as e:
                log.error(f"Error clearing embeddings: {e}")
                return {
                    "status": "partial_success",
                    "active_models": active_models,
                    "error": f"Model changed but failed to clear embeddings: {str(e)}"
                }
    else:
        config.LLM_CHAT_MODEL_NAME = req.name
        
    return {"status": "success", "active_models": active_models}

# Update the existing /models endpoint to include active status
@app.get("/models")
async def get_models_v2():
    import sys
    print("[DEBUG] ===== get_models_v2 CALLED =====", file=sys.stderr, flush=True)
    try:
        import urllib.request
        import json
        import os
        import sys
        
        # Get Ollama URL from config
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        ollama_url = f"{ollama_base_url}/api/tags"
        
        print(f"[DEBUG] Fetching models from Ollama at: {ollama_url}", file=sys.stderr, flush=True)
        
        # 1. Fetch models from Ollama
        models_list_raw = []
        try:
            req = urllib.request.Request(ollama_url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    models_list_raw = data.get("models", [])
                    print(f"[DEBUG] Successfully fetched {len(models_list_raw)} models from Ollama", file=sys.stderr, flush=True)
                else:
                    print(f"[ERROR] Ollama API returned status: {response.status}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to connect to Ollama at {ollama_url}: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
            # Don't fail completely, just return active models from config
            
        # 2. Get active models from config
        active = get_active_models()
        print(f"[DEBUG] Active models from config: {active}", file=sys.stderr, flush=True)
            
        # 3. Format for Frontend
        formatted_models = []
        for m in models_list_raw:
            model_name = m.get("name", "unknown")
            # Simple heuristic for type
            is_embedding = "embed" in model_name.lower() or "nomic" in model_name.lower() or "bert" in model_name.lower() or "mxbai" in model_name.lower()
            
            model_type = "embedding" if is_embedding else "llm"
            is_active = (model_type == "llm" and active.get("llm") in model_name) or \
                       (model_type == "embedding" and active.get("embedding") in model_name)
            
            formatted_models.append({
                "id": m.get("digest", model_name),
                "name": model_name,
                "type": model_type,
                "size": m.get("size", 0),
                "details": m.get("details", {}),
                "isActive": is_active
            })
        
        print(f"[DEBUG] Returning {len(formatted_models)} formatted models", file=sys.stderr, flush=True)
            
        return {
            "models": formatted_models,
            "active": active
        }
            
    except Exception as e:
        print(f"[ERROR] Error in get_models: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        active = get_active_models()
        return {"models": [], "active": active}


# ------------------------------------------------------------------------------
# Run the FastAPI server:
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("WARNING: Starting server without explicit uvicorn command. Not recommended for production use.")
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
