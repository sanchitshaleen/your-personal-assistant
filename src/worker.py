"""Celery configuration and task definitions for background processing."""

from celery import Celery
from celery.utils.log import get_task_logger
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Celery
app = Celery(
    'rag_server',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')
)

# Configure Celery settings
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='America/Chicago',  # CST/CDT timezone
    enable_utc=False,  # Use local timezone instead of UTC
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,  # Track when tasks start
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
)

# Get logger for tasks
logger = get_task_logger(__name__)


@app.task(bind=True, name='ingest_file_task', soft_time_limit=1800, time_limit=2000)
def ingest_file_task(self, user_id: str, file_name: str, collection_name: str = "documents", embedding_model: str = None, use_graph: bool = False):
    """Background task to ingest a file into the vector database and optionally Graph DB.
    
    Implements idempotency to prevent duplicate embeddings:
    - Checks if file is already embedded (status='completed')
    - If yes, skips re-embedding and returns early
    - Sets status to 'in_progress' before starting (prevents concurrent duplicates)
    - Clears old embeddings before re-ingesting (if status was failed or re-triggered)
    
    Args:
        user_id: User identifier
        file_name: Filename (with folder path if applicable, e.g. "test_1/file.pdf")
        collection_name: Qdrant collection name
        embedding_model: Embedding model name (optional)
        use_graph: Whether to identify entities and ingest into Neo4j Graph DB (Slower)
        
    Returns:
        dict: Status, document IDs, and message
    """
    sys.path.insert(0, '/fastAPI/src')
    
    from src.rag.indexer import ingest_file
    from src.rag.vector_store import VectorDB
    from src.rag.graph_indexer import GraphIndexer
    from src.core.llm import get_llm
    from config import settings as config
    from src.core import database as pg_db
    from src.processing import files
    
    # Construct file path using the files module (works in both Docker and local)
    file_path = files.get_file_path(user_id=user_id, file_name=file_name)
    
    # Use provided model or fallback to config
    model_name = embedding_model or config.EMB_MODEL_NAME
    
    try:
        # file_name already contains the relative path with folder if applicable
        # (e.g., "test_1/file.pdf" or just "file.pdf")
        logger.info(f"[TASK {self.request.id}] Getting file ID for '{file_name}' (user: {user_id})")
        
        file_id = pg_db.get_file_id_by_name(user_id=user_id, file_name=file_name)
        if file_id == -1:
            logger.error(f"[TASK {self.request.id}] File not found in database: {file_name}")
            return {
                'status': 'failed',
                'doc_ids': [],
                'message': f"File not found in database: {file_name}",
                'user_id': user_id
            }
        
        # IDEMPOTENCY CHECK: If already completed, skip re-embedding
        current_status = pg_db.get_embedding_status(file_id)
        if current_status == 'completed':
            logger.info(f"[TASK {self.request.id}] File {file_id} already fully embedded (status=completed). Skipping.")
            return {
                'status': 'success',
                'doc_ids': [],
                'message': f"File already embedded. Skipping duplicate embedding.",
                'user_id': user_id,
                'file_name': file_name,
                'skipped': True
            }
        
        # LOCK: Set status to 'in_progress' to prevent concurrent duplicate tasks
        logger.info(f"[TASK {self.request.id}] Setting file {file_id} embedding status to 'in_progress'")
        if not pg_db.set_embedding_status(file_id, 'in_progress'):
            logger.error(f"[TASK {self.request.id}] Failed to set embedding status")
            return {
                'status': 'failed',
                'doc_ids': [],
                'message': "Failed to set embedding status",
                'user_id': user_id
            }
        
        # If status was 'failed' or 'pending', clear old embeddings first
        if current_status in ['failed', 'pending']:
            logger.info(f"[TASK {self.request.id}] Clearing old embeddings for file {file_id} (previous status: {current_status})")
            pg_db.clear_embeddings_for_file(file_id)
        
        # Update task state to "PROGRESS"
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing...'}
        )
        logger.info(f"[TASK {self.request.id}] Starting ingestion for user '{user_id}', file '{file_path}'")
        
        # Initialize VectorDB
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': f'Initializing vector database with {model_name}...'}
        )
        vector_db = VectorDB(
            embed_model=model_name,
            retriever_num_docs=config.DOCS_NUM_COUNT,
            verify_connection=config.VERIFY_EMB_CONNECTION,
            qdrant_host=config.QDRANT_HOST,
            qdrant_port=config.QDRANT_PORT,
            collection_name=collection_name,
        )
        
        # Run ingestion
        self.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Processing file...'}
        )
        logger.info(f"[TASK {self.request.id}] Running ingest_file for '{file_path}'")
        
        
        # Initialize LLM for summarization (Common capability)
        llm_summary = get_llm(
            model_name=config.LLM_SUMMARY_MODEL_NAME, 
            context_size=8192, 
            temperature=config.LLM_SUMMARY_TEMPERATURE
        )

        # === EXCLUSIVE PATH SELECTION ===
        if use_graph and config.GRAPH_INGESTION_ENABLED:
            logger.info(f"[TASK {self.request.id}] EXCLUSIVE MODE: Graph DB selected. Skipping Qdrant ingestion.")
            
            # 1. Load Documents (Reusing logic for Graph Path)
            from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader
            from src.utils.splitter import split_text 
            # Note: We still use splitter to get manageable chunks for Graph, 
            # or we can pass full docs to LLMGraphTransformer (which chunks internally? No, better to chunk first).
            
            docs = []
            lower_path = file_path.lower()
            try:
                if lower_path.endswith(".pdf"):
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                elif lower_path.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(file_path)
                    docs = loader.load()
                elif lower_path.endswith(".txt"):
                    loader = TextLoader(file_path)
                    docs = loader.load()
                
                if not docs:
                     raise ValueError("No content loaded from file")
                     
            except Exception as load_e:
                logger.error(f"Failed to load file for Graph: {load_e}")
                return {
                    'status': 'failed',
                    'doc_ids': [],
                    'message': f"Failed to load file: {load_e}",
                    'user_id': user_id
                }

            # 2. Generate Summary (Independent of Qdrant)
            summary_text = None
            try:
                full_text = " ".join([d.page_content for d in docs])[:20000]
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                
                prompt = ChatPromptTemplate.from_template(
                    "Summarize the following document content in 3-5 sentences. "
                    "Focus on the main topics, entities, and purpose of the document.\n\n"
                    "Content:\n{content}"
                )
                chain = prompt | llm_summary | StrOutputParser()
                summary_text = chain.invoke({"content": full_text})
                logger.info(f"Generated summary for Graph file: {summary_text}")
            except Exception as sum_e:
                logger.warning(f"Failed to generate summary: {sum_e}")

            # 3. Ingest into Neo4j (Graph Only)
            try:
                 # Init LLM for extraction
                llm_chat = get_llm(
                    model_name=config.GRAPH_LLM_MODEL_NAME, 
                    context_size=8192, 
                    temperature=0, 
                    verify_connection=False
                )
                
                # GraphIndexer - using embeddings from VectorDB config (even if not using Qdrant DB)
                # We need embeddings object to embed text within Neo4j
                embeddings = vector_db.get_embeddings()
                
                graph_indexer = GraphIndexer(
                    llm=llm_chat,
                    embeddings=embeddings
                )
                
                self.update_state(
                    state='PROGRESS',
                    meta={'current': 60, 'total': 100, 'status': 'Ingesting into Neo4j Knowledge Graph...'}
                )
                
                # Use LangChain splitter to chunk documents for graph ingestion
                # LLMGraphTransformer is slow on large docs, better to chunk first.
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.DOC_CHAR_LIMIT,
                    chunk_overlap=config.DOC_OVERLAP_NO
                )
                split_docs = splitter.split_documents(docs)
                logger.info(f"Split into {len(split_docs)} chunks for graph extraction")
                
                # Pass parameters for summary creation
                success = graph_indexer.index_documents(
                    split_docs,
                    llm_summary=llm_summary,
                    qdrant_client=vector_db.db.client,  # Get Q drant client from VectorDB
                    user_id=user_id,
                    file_path=file_path
                )
                
                if not success:
                    raise Exception("Graph Indexer returned False")
                    
                doc_ids = ["graph_ingested"] # Placeholder as we don't track chunk IDs same way
                status = True
                message = "Ingested into Neo4j Graph successfully with document summary."
                
            except Exception as graph_e:
                logger.error(f"Graph Ingestion fatal error: {graph_e}")
                status = False
                message = f"Graph Ingestion failed: {graph_e}"
                doc_ids = []

        else:
            # === STANDARD PATH (Qdrant) ===
            logger.info(f"[TASK {self.request.id}] STANDARD MODE: Hybrid (Qdrant) ingestion.")
            status, doc_ids, message, summary_text = ingest_file(
                user_id=user_id,
                file_path=file_path,
                vectorstore=vector_db,
                embeddings=vector_db.get_embeddings(),
                llm_summary=llm_summary
            )
        
        if status:
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={'current': 80, 'total': 100, 'status': 'Storing metadata...'}
            )
            
            
            # Store embedding metadata in PostgreSQL using ATOMIC update
            logger.info(f"[TASK {self.request.id}] Storing metadata in PostgreSQL for file {file_id}")
            
            pg_db.update_file_success(
                file_id=file_id, 
                doc_ids=doc_ids, 
                summary=summary_text
            )
            
            # Set ingestion target based on mode
            try:
                ingestion_target = 'neo4j' if (use_graph and config.GRAPH_INGESTION_ENABLED) else 'qdrant'
                with pg_db.get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE uploads 
                        SET ingestion_target = %s
                        WHERE file_id = %s
                    """, (ingestion_target, file_id))
                    conn.commit()
                logger.info(f"[TASK {self.request.id}] Set ingestion_target={ingestion_target} for file {file_id}")
            except Exception as e:
                logger.error(f"Failed to set ingestion_target: {e}")
            
            # Final update
            self.update_state(
                state='PROGRESS',
                meta={'current': 100, 'total': 100, 'status': 'Complete!'}
            )
            
            logger.info(f"[TASK {self.request.id}] Ingestion completed successfully for '{file_name}'")
            return {
                'status': 'success',
                'doc_ids': doc_ids,
                'message': message,
                'user_id': user_id,
                'file_name': file_name
            }
        else:
            # Mark embedding as failed and store error message
            pg_db.set_embedding_status(file_id, 'failed')
            
            # Store error message in uploads table
            try:
                with pg_db.get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE uploads 
                        SET error_message = %s
                        WHERE file_id = %s
                    """, (str(message)[:500], file_id))  # Truncate to 500 chars
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to store error message: {e}")
            
            logger.error(f"[TASK {self.request.id}] Ingestion failed: {message}")
            return {
                'status': 'failed',
                'doc_ids': [],
                'message': message,
                'user_id': user_id
            }
            
    except Exception as e:
        # Try to mark embedding as failed even on exception
        try:
            file_id = pg_db.get_file_id_by_name(user_id=user_id, file_name=os.path.basename(file_path))
            if file_id != -1:
                pg_db.set_embedding_status(file_id, 'failed')
        except Exception as ex:
            logger.error(f"[TASK {self.request.id}] Failed to mark embedding status as failed: {ex}")
        
        logger.exception(f"[TASK {self.request.id}] Exception during ingestion: {e}")
        return {
            'status': 'error',
            'doc_ids': [],
            'message': f"Error during ingestion: {str(e)}",
            'user_id': user_id
        }


@app.task(bind=True, name='get_ingestion_status')
def get_ingestion_status(self, task_id: str):
    """Get the status of an ingestion task.
    
    Args:
        task_id: Celery task ID
        
    Returns:
        dict: Task status and progress information
    """
    result = app.AsyncResult(task_id)
    
    if result.state == 'PENDING':
        return {
            'status': 'pending',
            'progress': 0,
            'message': 'Task is waiting to be processed...'
        }
    elif result.state == 'PROGRESS':
        return {
            'status': 'processing',
            'progress': result.info.get('current', 0),
            'total': result.info.get('total', 100),
            'message': result.info.get('status', 'Processing...')
        }
    elif result.state == 'SUCCESS':
        return {
            'status': 'completed',
            'progress': 100,
            'message': 'Ingestion completed!',
            'result': result.result
        }
    elif result.state == 'FAILURE':
        return {
            'status': 'failed',
            'progress': 0,
            'message': f'Task failed: {str(result.info)}'
        }
    else:
        return {
            'status': result.state.lower(),
            'progress': 0,
            'message': f'Task state: {result.state}'
        }


if __name__ == '__main__':
    app.start()
