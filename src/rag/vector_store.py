""" Database Module for LLM System
- Contains the `VectorDB` class to manage a vector database using Qdrant and Ollama embeddings.
- Provides methods to initialize the database, retrieve embeddings, and perform similarity searches.
"""

import os
import time
import requests
from typing import Tuple, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import ConfigurableField

# For type hinting
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores import VectorStoreRetriever

from src.core.logger import get_logger
from src.rag.retrieval import BM25SemanticHybridRetriever

log = get_logger(name="core_database")


class VectorDB:
    """A class to manage the vector database using Qdrant and Ollama embeddings.

    Args:
        embed_model (str): The name of the Ollama embeddings model to use.
        retriever_num_docs (int): Number of documents to retrieve for similarity search.
        verify_connection (bool): Whether to verify the connection to the embeddings model.
        qdrant_host (str): Qdrant server host. Defaults to localhost.
        qdrant_port (int): Qdrant server port. Defaults to 6333.
        collection_name (str): Name of the Qdrant collection. Defaults to "documents".

    ## Functions:
        + `get_embeddings()`: Returns the Ollama embeddings model.
        + `get_vector_store()`: Returns the Qdrant vector store.
        + `get_retriever()`: Returns the retriever configured for similarity search.
    """

    def __init__(
        self, embed_model: str,
        retriever_num_docs: int = 5,
        verify_connection: bool = False,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "documents"
    ):
        self.retriever_num_docs = retriever_num_docs
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

        log.info(
            f"Initializing VectorDB with Qdrant at '{self.qdrant_url}', "
            f"embeddings='{embed_model}', collection='{collection_name}', k={retriever_num_docs} docs."
        )

        # Load embeddings from Ollama with Apple Silicon optimization
        # On Apple Silicon (M1/M2/M3), Ollama automatically uses Metal for GPU acceleration
        # Setting num_gpu=-1 tells Ollama to use all available GPU cores
        self.embeddings = OllamaEmbeddings(
            model=embed_model, 
            num_gpu=-1,  # Use all available GPU cores on Apple Silicon
            keep_alive=-1,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")  # Use config/env variable
        )

        if verify_connection:
            try:
                self.embeddings.embed_documents(['a'])
                log.info(f"Embeddings model '{embed_model}' initialized and verified.")

            except Exception as e:
                log.error(f"Failed to initialize Embeddings: {e}")
                raise RuntimeError(f"Couldn't initialize Embeddings model '{embed_model}'") from e
        else:
            log.warning(f"Embeddings '{embed_model}' initialized without connection verification.")

        # Initialize Qdrant vector store with retry logic
        max_retries = 10
        retry_delay = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Try to connect to existing collection first
                try:
                    self.db = Qdrant(
                        url=self.qdrant_url,
                        collection_name=self.collection_name,
                        embeddings=self.embeddings,
                        prefer_grpc=False,  # Use HTTP for better compatibility
                    )
                    log.info(f"Connected to existing Qdrant collection '{self.collection_name}'.")
                except Exception as e:
                    # Collection doesn't exist, need to create it
                    # Create via HTTP API directly for better control
                    log.info(f"Collection '{self.collection_name}' doesn't exist, creating empty one...")
                    
                    try:
                        # Create empty collection via HTTP API
                        # Determine vector size based on embedding model
                        vector_size = len(self.embeddings.embed_query("test"))
                        log.info(f"Detected embedding dimension: {vector_size}")
                        
                        create_url = f"{self.qdrant_url}/collections/{self.collection_name}"
                        payload = {
                            "vectors": {
                                "size": vector_size,
                                "distance": "Cosine"
                            }
                        }
                        response = requests.put(create_url, json=payload, timeout=10)
                        if response.status_code == 200:
                            log.info(f"Created empty Qdrant collection '{self.collection_name}' via HTTP API.")
                            # Now connect to the created collection
                            self.db = Qdrant(
                                url=self.qdrant_url,
                                collection_name=self.collection_name,
                                embeddings=self.embeddings,
                                prefer_grpc=False,
                            )
                        else:
                            raise Exception(f"Failed to create collection: {response.text}")
                    except Exception as create_error:
                        log.warning(f"HTTP API creation failed, trying fallback with dummy doc: {create_error}")
                        # Fallback: create with minimal dummy document if HTTP fails
                        dummy_doc = Document(
                            page_content=".",  # Single character to avoid empty content issues
                            metadata={"user_id": "system", "source": "initialization", "skip_in_search": True}
                        )
                        self.db = Qdrant.from_documents(
                            [dummy_doc],
                            self.embeddings,
                            url=self.qdrant_url,
                            collection_name=self.collection_name,
                            prefer_grpc=False,
                        )
                        log.info(f"Created Qdrant collection '{self.collection_name}' with minimal dummy doc.")
                break
            except Exception as e:
                last_error = e
                log.warning(f"Qdrant connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    log.error(f"Failed to initialize Qdrant after {max_retries} attempts")
                    raise RuntimeError(f"Couldn't connect to Qdrant at {self.qdrant_url}") from last_error

        # Create configurable retriever for runtime filter application
        retriever = self.db.as_retriever(
            search_kwargs={"k": retriever_num_docs}
        )
        configurable_retriever = retriever.configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
                name="Search Kwargs",
                description="The search kwargs to use",
            )
        )

        self.retriever = configurable_retriever
        log.info(f"Created configurable retriever with k={retriever_num_docs}.")

    def get_embeddings(self) -> Embeddings:
        log.info("Returning the Embeddings model instance.")
        return self.embeddings

    def get_vector_store(self) -> VectorStore:
        log.info("Returning the Qdrant vector store instance.")
        return self.db

    def get_retriever(self) -> VectorStoreRetriever:
        log.info("Returning the retriever for similarity search.")
        log.info(f"Retriever search_kwargs: {self.retriever.search_kwargs if hasattr(self.retriever, 'search_kwargs') else 'N/A'}")
        return self.retriever  # type: ignore[return-value]
    
    def get_bm25_semantic_hybrid_retriever(
        self, 
        bm25_weight: float = 0.5, 
        semantic_weight: float = 0.5
    ) -> BM25SemanticHybridRetriever:
        """Get a deterministic hybrid retriever combining BM25 and Semantic search.
        
        This hybrid approach is simpler and more deterministic than SPLADE:
        - BM25: Sparse lexical matching (deterministic, no floating-point variations)
        - Semantic: Dense vector search (deterministic cosine similarity)
        - Fusion: Weighted combination (fully deterministic)
        
        Args:
            bm25_weight: Weight for BM25 lexical scores (default 0.5)
            semantic_weight: Weight for semantic vector scores (default 0.5)
        
        Returns:
            BM25SemanticHybridRetriever instance
            
        Benefits:
            - DETERMINISTIC: Same query always returns same chunk order
            - HYBRID: Combines lexical and semantic understanding
            - SIMPLE: No external APIs or complex fusion methods
            - RELIABLE: Weighted average of two stable scoring methods
        """
        log.info(f"Creating BM25+Semantic hybrid retriever: BM25_weight={bm25_weight}, Semantic_weight={semantic_weight}")
        
        # Get Qdrant client from the vector store
        qdrant_client = self.db.client
        
        # Create BM25 + Semantic hybrid retriever
        hybrid = BM25SemanticHybridRetriever(
            qdrant_client=qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            k=self.retriever_num_docs
        )
        
        return hybrid

    def delete_documents(self, document_ids: list) -> int:
        """Delete documents from Qdrant by their IDs.
        
        Args:
            document_ids: List of document IDs (as strings) to delete
            
        Returns:
            Number of documents deleted
        """
        if not document_ids:
            log.warning("No document IDs provided for deletion")
            return 0
            
        try:
            # Convert string IDs to integers for Qdrant
            vector_ids_int = [int(doc_id) for doc_id in document_ids]
            
            log.info(f"Deleting {len(vector_ids_int)} documents from Qdrant collection '{self.collection_name}'")
            
            # Delete from Qdrant using the client
            self.db.client.delete(
                collection_name=self.collection_name,
                points_selector=vector_ids_int
            )
            
            log.info(f"Successfully deleted {len(vector_ids_int)} documents from Qdrant")
            return len(vector_ids_int)
            
        except Exception as e:
            log.error(f"Error deleting documents from Qdrant: {e}", exc_info=True)
            raise

