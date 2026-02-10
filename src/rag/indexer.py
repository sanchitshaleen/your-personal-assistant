""" A script which will deal with ingestion of new documents into the vector database.
- Currently has file ingestion which supports txt, pdf, and md files.
- Plan to add more file types in the future.
- Plan to add web based ingestion in the future.
"""

from typing import List

from src.utils.loader import load_file
from src.utils.splitter import split_text

# For type hinting
from src.rag.vector_store import VectorDB
from langchain_core.embeddings import Embeddings


from src.core.logger import get_logger
log = get_logger(name="core_ingestion")


def ingest_file(user_id: str, file_path: str, vectorstore: VectorDB,
                embeddings: Embeddings, llm_summary=None) -> tuple[bool, List[str], str, str]:
    """Ingest a file into the vector database. Returns the ids of vector embeddings stored in database.

    Args:
        file_path (str): The absolute path to the file to be ingested.
        db (VectorDB): The vector database instance.
        embeddings (Embeddings): The embeddings model to use for the documents.
        llm_summary: Optional LLM model for summary generation.

    Returns:
        tuple[bool, List[str], str, str]: status, doc_ids, message, summary_text
    """

    # Load the file and get its content as Document objects:
    status, documents, message = load_file(user_id, file_path)
    # print(status, documents, message)

    if not status:
        return False, [], message

    # Split the documents into smaller chunks:
    status, split_docs, message = split_text(documents)
    if status and not split_docs:
        log.warning(f"No content found in the file: {file_path}")
        return True, [], f"No content found in the file: {file_path}"

    if not status:
        return False, [], message

    # Add the split documents to the vector database:
    try:
        # === NEW: GENERATE AND EMBED SUMMARY ===
        summary_text = None
        if llm_summary:
            try:
                log.info(f"Generating summary for file: {file_path}...")
                # Create a simple chain for summarization
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                
                # Combine first 20000 chars roughly as context for summary
                full_text = " ".join([d.page_content for d in documents])[:20000]
                
                prompt = ChatPromptTemplate.from_template(
                    "Summarize the following document content in 3-5 sentences. "
                    "Focus on the main topics, entities, and purpose of the document so a retrieval system knows when to select it.\n\n"
                    "Content:\n{content}"
                )
                chain = prompt | llm_summary | StrOutputParser()
                summary_text = chain.invoke({"content": full_text})
                log.info(f"Generated summary: {summary_text}")
                
                # Embed successful summary
                if summary_text:
                    log.info("Embedding summary for Planner Agent...")
                    summary_embedding = embeddings.embed_query(summary_text)
                    
                    # Store in Qdrant 'document_summaries' collection
                    from qdrant_client.models import PointStruct
                    summary_point = PointStruct(
                        id=int(time.time() * 1000000), # Simple unique ID
                        vector=summary_embedding,
                        payload={
                            "page_content": summary_text,
                            "file_path": file_path,
                            "user_id": user_id,
                            "filename": file_path.split("/")[-1]
                        }
                    )
                    
                    # Ensure collection exists
                    try:
                        vectorstore.db.client.upsert(
                            collection_name="document_summaries",
                            points=[summary_point]
                        )
                    except Exception:
                        # Create if doesn't exist
                        try:
                            vectorstore.db.client.recreate_collection(
                                collection_name="document_summaries",
                                vectors_config=vectorstore.db.client.get_collection(vectorstore.collection_name).config.params.vectors
                            )
                            vectorstore.db.client.upsert(
                                collection_name="document_summaries",
                                points=[summary_point]
                            )
                        except Exception as e:
                            log.error(f"Failed to create/upsert document_summaries: {e}")

            except Exception as e:
                log.error(f"Failed to generate/store summary: {e}")


        log.info(f"Ingesting {len(split_docs)} documents with dense and sparse embeddings...")
        
        # CRITICAL FIX: Generate BOTH dense and sparse embeddings explicitly
        # Dense embeddings: mxbai-embed-large (1024-dim vectors)
        # Sparse embeddings: SPLADE via FastEmbed (keyword-based sparse vectors for hybrid search)
        
        texts = [doc.page_content for doc in split_docs]
        
        # Get max safe embedding length dynamically from model's context limit
        try:
            from src.utils.model_utils import get_max_embed_length
            from config.settings import EMB_MODEL_NAME
            MAX_EMBED_LENGTH = get_max_embed_length(EMB_MODEL_NAME)
        except Exception as e:
            # Fallback: very conservative estimate (200 tokens * 4 chars/token * 0.9 safety)
            log.warning(f"Could not get dynamic max embed length: {e}. Using fallback value.")
            MAX_EMBED_LENGTH = 720
        
        # Safety check: Truncate any text that exceeds the embedding model's context limit
        texts = [text[:MAX_EMBED_LENGTH] if len(text) > MAX_EMBED_LENGTH else text for text in texts]
        
        # Log if any texts were truncated
        truncated_count = sum(1 for doc, text in zip(split_docs, texts) if len(doc.page_content) > MAX_EMBED_LENGTH)
        if truncated_count > 0:
            log.warning(f"Truncated {truncated_count} documents to {MAX_EMBED_LENGTH} chars to fit embedding model context limit")
        
        # 1. Generate DENSE embeddings with batching to avoid context length exceeded errors
        log.info(f"Generating dense embeddings for {len(texts)} document chunks...")
        embedding_vectors = []
        batch_size = 32  # Increased batch size for faster processing with local Ollama
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            log.info(f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
            try:
                batch_vectors = embeddings.embed_documents(batch_texts)
                embedding_vectors.extend(batch_vectors)
            except Exception as batch_error:
                log.error(f"Error embedding batch {i//batch_size + 1}: {batch_error}")
                raise
        log.info(f"Successfully generated {len(embedding_vectors)} dense embeddings.")
        log.info(f"Dense embedding dimensions: {len(embedding_vectors[0])}")
        
        # 2. Generate SPARSE embeddings (SPLADE) for hybrid retrieval (SKIP for small files)
        sparse_vectors_list = None
        # Only generate SPLADE for files with 5+ chunks to save time on small files
        if len(texts) >= 5:
            try:
                from fastembed import SparseTextEmbedding
                log.info("Initializing SPLADE sparse embedding model from FastEmbed...")
                sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
                
                # Generate sparse embeddings with batching
                log.info(f"Generating SPLADE sparse embeddings for {len(texts)} documents...")
                sparse_vectors_list = []
                sparse_batch_size = 32  # Increased batch size for faster SPLADE processing
                for i in range(0, len(texts), sparse_batch_size):
                    batch_texts = texts[i:i + sparse_batch_size]
                    log.info(f"SPLADE embedding batch {i//sparse_batch_size + 1}/{(len(texts) + sparse_batch_size - 1)//sparse_batch_size}...")
                    try:
                        batch_sparse = list(sparse_model.embed(batch_texts))
                        sparse_vectors_list.extend(batch_sparse)
                    except Exception as sparse_batch_error:
                        log.warning(f"Error generating SPLADE batch {i//sparse_batch_size + 1}: {sparse_batch_error}")
                        # Don't fail the entire ingestion if SPLADE fails
                        sparse_vectors_list = None
                        break
                log.info(f"Successfully generated {len(sparse_vectors_list)} sparse embeddings (SPLADE).")
                
            except ImportError:
                log.warning("fastembed package not found or SparseTextEmbedding not available. SPLADE sparse vectors will be skipped.")
                sparse_vectors_list = None
            except Exception as e:
                log.warning(f"Failed to generate SPLADE embeddings: {e}. Continuing with dense embeddings only.")
                sparse_vectors_list = None
        else:
            log.info(f"File has {len(texts)} chunks (< 5). Skipping SPLADE sparse embeddings for faster processing.")
        
        # 3. Add documents to Qdrant with dense vectors
        from qdrant_client.models import PointStruct
        import time
        
        points_to_add = []
        for idx, (doc, dense_vector) in enumerate(zip(split_docs, embedding_vectors)):
            # Generate unique point ID using timestamp + index for guaranteed uniqueness
            point_id_int = int(time.time() * 1000000) + idx  # microsecond precision + index
            # Ensure it's within unsigned 64-bit range
            point_id_int = point_id_int % (2**63)
            
            # Build payload with document content and metadata
            payload = {
                "page_content": doc.page_content,
                **doc.metadata,  # Include all metadata from document
            }
            
            # Add SPLADE sparse vector to payload if available
            if sparse_vectors_list and idx < len(sparse_vectors_list):
                sparse_vec = sparse_vectors_list[idx]
                # FastEmbed's SparseEmbedding has .indices and .values attributes
                try:
                    payload["sparse_indices"] = sparse_vec.indices.tolist() if hasattr(sparse_vec.indices, 'tolist') else list(sparse_vec.indices)
                    payload["sparse_values"] = sparse_vec.values.tolist() if hasattr(sparse_vec.values, 'tolist') else list(sparse_vec.values)
                    log.debug(f"Doc {idx}: SPLADE sparse vector with {len(payload['sparse_indices'])} non-zero dimensions")
                except Exception as e:
                    log.debug(f"Could not extract SPLADE data for doc {idx}: {e}")
            
            # Create point with dense vector (required for vector search)
            point = PointStruct(
                id=point_id_int,
                vector=dense_vector,
                payload=payload
            )
            points_to_add.append(point)
        
        # Upsert all points into Qdrant collection using the vectorstore's client
        # This ensures the collection exists and is properly configured
        log.info(f"Upserting {len(points_to_add)} points to Qdrant collection '{vectorstore.collection_name}'...")
        
        try:
            vectorstore.db.client.upsert(
                collection_name=vectorstore.collection_name,
                points=points_to_add
            )
        except Exception as upsert_error:
            # If collection doesn't exist, create it via LangChain's from_documents which properly initializes it
            log.warning(f"Upsert failed (collection may not exist): {upsert_error}")
            log.info("Creating collection via from_documents with first document...")
            
            # Use from_documents to create and initialize the collection properly
            if len(split_docs) > 0:
                from langchain_community.vectorstores import Qdrant as QdrantVectorStore
                try:
                    # Create collection with first document - this initializes everything correctly
                    qdrant_store = QdrantVectorStore.from_documents(
                        [split_docs[0]],
                        embedding=vectorstore.embeddings,
                        url=vectorstore.qdrant_url,
                        collection_name=vectorstore.collection_name,
                        prefer_grpc=False,
                    )
                    log.info(f"Created collection via from_documents. Now upserting remaining {len(points_to_add)} points...")
                    
                    # Now upsert all points (including the first one which will be updated)
                    vectorstore.db.client.upsert(
                        collection_name=vectorstore.collection_name,
                        points=points_to_add
                    )
                except Exception as create_error:
                    log.error(f"Failed to create collection: {create_error}")
                    raise
        
        doc_ids = [str(p.id) for p in points_to_add]
        
        splade_status = f" + SPLADE sparse vectors" if sparse_vectors_list else ""
        log.info(f"Successfully added {len(split_docs)} documents with dense embeddings{splade_status} to Qdrant.")
        return True, doc_ids, f"Ingested {len(split_docs)} documents successfully.", summary_text
            
    except Exception as e:
        log.error(f"Failed to ingest documents into Qdrant: {e}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return False, [], f"Failed to ingest documents: {e}", None


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain.callbacks.tracers.langchain import wait_for_all_tracers
    load_dotenv()

    # Example usage
    user = "test_user"
    # example_file_path = "../../../GenAI/Data/attention_is_all_you_need_1706.03762v7.pdf"
    example_file_path = "../../../GenAI/Data/speech.md"

    vector_db = VectorDB(
        embed_model="mxbai-embed-large:latest",
        qdrant_host="localhost",
        qdrant_port=6333
    )

    status, doc_ids, message = ingest_file(user, example_file_path, vector_db, vector_db.embeddings)
    if status:
        print(doc_ids)
    else:
        print(f"Error: {message}")

    # Retrieve the documents to verify ingestion:
    print(
        vector_db.retriever.invoke(
            input="What is the attention mechanism in transformers?",
            config={"configurable": {
                "search_kwargs": {
                    "k": 3,
                    "filter": {"user_id": user}
                }
            }}
        )
    )

    print(
        vector_db.retriever.invoke(
            input="What is the attention mechanism in transformers?",
            filter={"user_id": "random"}
        )
    )

    wait_for_all_tracers()
