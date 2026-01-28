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
def ingest_file_task(self, user_id: str, file_name: str, collection_name: str = "documents", embedding_model: str = None):
    """Background task to ingest a file into the vector database.
    
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
        
    Returns:
        dict: Status, document IDs, and message
    """
    sys.path.insert(0, '/fastAPI/src')
    
    from src.rag.indexer import ingest_file
    from src.rag.vector_store import VectorDB
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
        
        status, doc_ids, message = ingest_file(
            user_id=user_id,
            file_path=file_path,
            vectorstore=vector_db,
            embeddings=vector_db.get_embeddings()
        )
        
        if status:
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={'current': 80, 'total': 100, 'status': 'Storing metadata...'}
            )
            
            # Store embedding metadata in PostgreSQL
            logger.info(f"[TASK {self.request.id}] Storing {len(doc_ids)} embeddings in PostgreSQL for file {file_id}")
            
            for vid in doc_ids:
                pg_db.add_embedding(file_id=file_id, vector_id=vid)
            
            # Update chunk count
            pg_db.update_file_chunk_count(file_id=file_id, chunk_count=len(doc_ids), user_id=user_id)
            
            # Mark file as available (embeddings ready for querying)
            logger.info(f"[TASK {self.request.id}] Marking file as available after embedding")
            try:
                # Update the uploads table to mark file as available
                import psycopg2
                with pg_db.get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE uploads SET available = 1
                        WHERE file_id = %s AND user_id = %s
                    """, (file_id, user_id))
                    conn.commit()
                    logger.info(f"[TASK {self.request.id}] File marked as available: {file_id}")
            except Exception as e:
                logger.error(f"[TASK {self.request.id}] Error marking file as available: {e}")
            
            # Mark embedding as completed
            pg_db.set_embedding_status(file_id, 'completed')
            
            # Final update
            self.update_state(
                state='PROGRESS',
                meta={'current': 100, 'total': 100, 'status': 'Complete!'}
            )
            
            logger.info(f"[TASK {self.request.id}] Ingestion completed successfully for '{file_name}' ({len(doc_ids)} embeddings)")
            return {
                'status': 'success',
                'doc_ids': doc_ids,
                'message': f"Ingested {len(doc_ids)} documents successfully.",
                'user_id': user_id,
                'file_name': file_name
            }
        else:
            # Mark embedding as failed
            pg_db.set_embedding_status(file_id, 'failed')
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
