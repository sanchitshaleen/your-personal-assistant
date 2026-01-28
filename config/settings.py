"""
config.py - Central configuration for AI System.

This module stores all configurable constants related to:
- LLMs (chat, summarization)
- Embeddings
- Chunking and content limits
- Verification checks
- Verification checks
- Qdrant Vector Database

DYNAMIC CONFIGURATION:
Most values can be dynamically calculated based on actual model capabilities.
This allows easy model switching without manual configuration updates.
"""

import os

# Model configuration::
LLM_CHAT_MODEL_NAME: str = "gemma3:1b"                # Chatting model (1B - balanced speed/quality)
LLM_CHAT_TEMPERATURE: float = 0.75
LLM_SUMMARY_MODEL_NAME: str = "gemma3:1b"             # History Summarization model
LLM_SUMMARY_TEMPERATURE: float = 0.5
EMB_MODEL_NAME: str = "nomic-embed-text"           # Embeddings model (using latest tag)


# Verification configuration:
#   - Whether to immediately verify the connection to
#   - the LLM models and the Embeddings models after initialization.
#   - Useful specifically for Ollama models, as they can be loaded on GPU or CPU.
VERIFY_LLM_CONNECTION: bool = False
VERIFY_EMB_CONNECTION: bool = False


# ============================================================================
# DYNAMIC CONFIGURATION SECTION
# ============================================================================
# When enabled, these values are calculated at runtime based on actual model
# capabilities. This makes model switching seamless and automatic.

# Enable dynamic configuration based on model capabilities
ENABLE_DYNAMIC_CONFIG: bool = os.getenv("ENABLE_DYNAMIC_CONFIG", "true").lower() == "true"

# Ollama API base URL for model capability detection
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


# ============================================================================
# DOCUMENT CHUNKING PROPERTIES
# ============================================================================

# Default values (used if dynamic calculation is disabled)
DOC_CHAR_LIMIT: int = 800                               # Char limit for each doc (reduced from 1600 for safety)
DOC_OVERLAP_NO: int = 80                                # Char limit for chunk overlap (reduced from 160)

# Dynamic chunk sizing based on embedding model's context length
ENABLE_DYNAMIC_CHUNK_SIZE: bool = os.getenv("ENABLE_DYNAMIC_CHUNK_SIZE", "true").lower() == "true"
CHUNK_SIZE_SAFETY_MARGIN: float = float(os.getenv("CHUNK_SIZE_SAFETY_MARGIN", "0.7"))  # More conservative (reduced from 0.8)


# ============================================================================
# DOCUMENT RETRIEVAL PROPERTIES
# ============================================================================

# Default values (used if dynamic calculation is disabled)
DOC_TOKEN_SIZE: int = DOC_CHAR_LIMIT // 4               # Appx number of tokens in each doc
DOCS_NUM_COUNT: int = 3                                 # Default number of docs to retrieve

# Dynamic retrieval sizing based on LLM's context length
ENABLE_DYNAMIC_RETRIEVAL_SIZE: bool = os.getenv("ENABLE_DYNAMIC_RETRIEVAL_SIZE", "true").lower() == "true"
RETRIEVAL_SAFETY_MARGIN: float = float(os.getenv("RETRIEVAL_SAFETY_MARGIN", "0.6"))


# ============================================================================
# RAG CONTEXT SIZE (MAX_CONTENT_SIZE)
# ============================================================================

# Default maximum content size (tokens allowed for chat_history + input + context)
# This will be dynamically calculated based on LLM's context length if enabled
MAX_CONTENT_SIZE: int = 14000

# Enable dynamic MAX_CONTENT_SIZE calculation
ENABLE_DYNAMIC_MAX_CONTENT: bool = os.getenv("ENABLE_DYNAMIC_MAX_CONTENT", "true").lower() == "true"
MAX_CONTENT_SAFETY_MARGIN: float = float(os.getenv("MAX_CONTENT_SAFETY_MARGIN", "0.75"))


# ============================================================================
# QDRANT VECTOR DATABASE
# ============================================================================

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME: str = "documents"


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# BM25 + Semantic Hybrid Retrieval (Deterministic)
# Combines lexical matching (BM25) with semantic vector search
USE_BM25_SEMANTIC_HYBRID: bool = os.getenv("USE_BM25_SEMANTIC_HYBRID", "false").lower() == "true"
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.5"))                # Weight for BM25 scores
SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.5"))        # Weight for semantic scores


# ============================================================================
# SEMANTIC CACHING CONFIGURATION
# ============================================================================

SEMANTIC_CACHE_ENABLED: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.85"))
SEMANTIC_CACHE_TTL_SECONDS: int = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))  # 24 hours
SEMANTIC_CACHE_MAX_PER_USER: int = int(os.getenv("SEMANTIC_CACHE_MAX_PER_USER", "100"))


# ============================================================================
# RUNTIME CONFIGURATION INITIALIZATION
# ============================================================================
# These will be populated at runtime if ENABLE_DYNAMIC_CONFIG is True

_DYNAMIC_CONFIG_INITIALIZED = False

def _initialize_dynamic_config():
    """Initialize dynamic configuration based on actual model capabilities."""
    global _DYNAMIC_CONFIG_INITIALIZED
    global DOC_CHAR_LIMIT, DOC_OVERLAP_NO
    global DOCS_NUM_COUNT, DOC_TOKEN_SIZE
    global MAX_CONTENT_SIZE
    
    if _DYNAMIC_CONFIG_INITIALIZED or not ENABLE_DYNAMIC_CONFIG:
        return
    
    try:
        from src.utils.model_utils import (
            get_embedding_model_context_length,
            get_llm_context_length,
            calculate_optimal_chunk_size,
            calculate_optimal_max_content_size,
            calculate_optimal_docs_count
        )
        
        # Get embedding model context
        emb_context = get_embedding_model_context_length(EMB_MODEL_NAME, OLLAMA_BASE_URL)
        
        # Get LLM context
        llm_context = get_llm_context_length(LLM_CHAT_MODEL_NAME, OLLAMA_BASE_URL)
        
        # Calculate chunk sizes
        if ENABLE_DYNAMIC_CHUNK_SIZE:
            chunk_size, chunk_overlap = calculate_optimal_chunk_size(
                emb_context,
                safety_margin=CHUNK_SIZE_SAFETY_MARGIN
            )
            DOC_CHAR_LIMIT = chunk_size
            DOC_OVERLAP_NO = chunk_overlap
        
        # Calculate retrieval sizes
        if ENABLE_DYNAMIC_RETRIEVAL_SIZE:
            DOC_TOKEN_SIZE = DOC_CHAR_LIMIT // 4
            DOCS_NUM_COUNT = calculate_optimal_docs_count(
                llm_context,
                chunk_size_chars=DOC_CHAR_LIMIT,
                safety_margin=RETRIEVAL_SAFETY_MARGIN
            )
        
        # Calculate MAX_CONTENT_SIZE
        if ENABLE_DYNAMIC_MAX_CONTENT:
            MAX_CONTENT_SIZE = calculate_optimal_max_content_size(
                llm_context,
                chunk_size_chars=DOC_CHAR_LIMIT,
                max_docs_to_retrieve=DOCS_NUM_COUNT,
                safety_margin=MAX_CONTENT_SAFETY_MARGIN
            )
        
        _DYNAMIC_CONFIG_INITIALIZED = True
        
        import logging
        logger = logging.getLogger("config")
        logger.info(
            f"Dynamic configuration initialized:\n"
            f"  Chunk size: {DOC_CHAR_LIMIT} chars\n"
            f"  Chunk overlap: {DOC_OVERLAP_NO} chars\n"
            f"  Docs to retrieve: {DOCS_NUM_COUNT}\n"
            f"  MAX_CONTENT_SIZE: {MAX_CONTENT_SIZE} tokens"
        )
        
    except Exception as e:
        import logging
        logger = logging.getLogger("config")
        logger.warning(f"Failed to initialize dynamic config: {e}. Using static defaults.")
        _DYNAMIC_CONFIG_INITIALIZED = True
HYBRID_DENSE_LIMIT: int = 20                    # Number of dense search results before fusion
HYBRID_CANDIDATES_LIMIT: int = 100              # Candidate limit for MMR pre-filtering
HYBRID_SPLADE_BATCH_SIZE: int = 6               # Batch size for SPLADE document processing
HYBRID_RETRIEVER_K: int = DOCS_NUM_COUNT        # Number of final documents to retrieve

# Dummy response mode properties:
TOKENS_PER_SEC: int = 50                                # num of tokens yielded per sec
BATCH_TOKEN_PS: int = 2                                 # num of tokens yielded in each batch
