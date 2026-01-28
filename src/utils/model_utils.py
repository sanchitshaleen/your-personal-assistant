"""Embedding and LLM model utilities for dynamic configuration.

This module provides utilities to:
1. Detect embedding model's context length and capabilities dynamically
2. Detect LLM model's context length and capabilities dynamically
3. Calculate optimal chunk sizes based on context limits
4. Handle different embedding models with different constraints
5. Suggest optimal parameters based on model capabilities
"""

import requests
from typing import Optional, Tuple, Dict, Any
from src.core.logger import get_logger

log = get_logger(name="model_utils")


def get_model_info(
    model_name: str,
    ollama_base_url: str = "http://ollama:11434"
) -> Dict[str, Any]:
    """Get comprehensive information about a model from Ollama.
    
    Args:
        model_name: Name of the Ollama model
        ollama_base_url: Base URL of the Ollama API
        
    Returns:
        Dictionary containing model information (context_length, parameters, etc)
    """
    try:
        response = requests.post(
            f"{ollama_base_url}/api/show",
            json={"name": model_name},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            log.warning(f"Could not get info for model '{model_name}': HTTP {response.status_code}")
            return {}
            
    except requests.exceptions.RequestException as e:
        log.warning(f"Error querying Ollama for model '{model_name}': {e}")
        return {}
    except Exception as e:
        log.warning(f"Unexpected error getting model info: {e}")
        return {}


def get_embedding_model_context_length(
    model_name: str,
    ollama_base_url: str = "http://ollama:11434"
) -> int:
    """Get the context length of an embedding model from Ollama.
    
    Args:
        model_name: Name of the Ollama embedding model (e.g., "mxbai-embed-large:latest")
        ollama_base_url: Base URL of the Ollama API
        
    Returns:
        Context length in tokens. Returns 512 as fallback if unable to detect.
    """
    try:
        model_info = get_model_info(model_name, ollama_base_url)
        
        if "details" in model_info:
            details = model_info["details"]
            if "context_length" in details:
                ctx_len = details["context_length"]
                log.info(f"Model '{model_name}' context length: {ctx_len} tokens")
                return ctx_len
        
        # Alternative: check parameters
        if "parameters" in model_info:
            params = model_info["parameters"]
            if isinstance(params, dict) and "num_ctx" in params:
                ctx_len = params["num_ctx"]
                log.info(f"Model '{model_name}' context length (from num_ctx): {ctx_len} tokens")
                return ctx_len
        
        log.warning(f"Could not determine context length for '{model_name}', using default 512")
        return 512
        
    except Exception as e:
        log.warning(f"Error getting embedding model context length: {e}. Using default 512 tokens")
        return 512


def get_llm_context_length(
    model_name: str,
    ollama_base_url: str = "http://ollama:11434"
) -> int:
    """Get the context length of an LLM model from Ollama.
    
    Args:
        model_name: Name of the Ollama LLM model
        ollama_base_url: Base URL of the Ollama API
        
    Returns:
        Context length in tokens. Returns 4096 as fallback if unable to detect.
    """
    try:
        model_info = get_model_info(model_name, ollama_base_url)
        
        if "details" in model_info:
            details = model_info["details"]
            if "context_length" in details:
                ctx_len = details["context_length"]
                log.info(f"LLM '{model_name}' context length: {ctx_len} tokens")
                return ctx_len
        
        # Alternative: check parameters
        if "parameters" in model_info:
            params = model_info["parameters"]
            if isinstance(params, dict) and "num_ctx" in params:
                ctx_len = params["num_ctx"]
                log.info(f"LLM '{model_name}' context length (from num_ctx): {ctx_len} tokens")
                return ctx_len
        
        log.warning(f"Could not determine context length for LLM '{model_name}', using default 4096")
        return 4096
        
    except Exception as e:
        log.warning(f"Error getting LLM context length: {e}. Using default 4096 tokens")
        return 4096


def calculate_optimal_chunk_size(
    context_length_tokens: int,
    safety_margin: float = 0.8,
    chars_per_token: float = 4.0
) -> Tuple[int, int]:
    """Calculate optimal chunk and overlap sizes based on embedding model's context length.
    
    Args:
        context_length_tokens: Maximum context length the embedding model supports
        safety_margin: Fraction of context to actually use (e.g., 0.8 = use 80% to leave headroom)
        chars_per_token: Approximate characters per token (typically 3.5-4.5, avg 4)
        
    Returns:
        Tuple of (chunk_size_chars, overlap_chars)
        
    Example:
        For mxbai-embed-large with 512 tokens:
        - Safe token budget: 512 * 0.8 = 409 tokens
        - Safe char budget: 409 * 4 = 1636 chars
        Returns (1600, 160) for nice round numbers
    """
    # Calculate safe token budget with margin
    safe_tokens = int(context_length_tokens * safety_margin)
    
    # Convert to characters
    safe_chars = int(safe_tokens * chars_per_token)
    
    # Round to nice numbers for consistency
    chunk_size = (safe_chars // 100) * 100  # Round down to nearest 100
    
    # Overlap is typically 10% of chunk size
    overlap_size = max(chunk_size // 10, 50)  # At least 50 chars for overlap
    
    log.info(
        f"Calculated chunk sizes: chunk={chunk_size} chars ({chunk_size // 4} tokens), "
        f"overlap={overlap_size} chars. "
        f"(Model context: {context_length_tokens} tokens, safety margin: {int(safety_margin*100)}%)"
    )
    
    return chunk_size, overlap_size


def get_dynamic_chunk_config(
    model_name: str,
    ollama_base_url: str = "http://ollama:11434",
    safety_margin: float = 0.8
) -> Tuple[int, int]:
    """Get chunk configuration dynamically based on the embedding model.
    
    This is a convenience function that combines:
    1. Detecting the model's context length from Ollama
    2. Calculating optimal chunk sizes
    
    Args:
        model_name: Name of the Ollama embedding model
        ollama_base_url: Base URL of Ollama API
        safety_margin: Safety margin to apply (0.8 = use 80% of context)
        
    Returns:
        Tuple of (chunk_size_chars, overlap_chars)
    """
    context_length = get_embedding_model_context_length(model_name, ollama_base_url)
    return calculate_optimal_chunk_size(context_length, safety_margin)


def get_max_embed_length(
    model_name: str,
    ollama_base_url: str = "http://ollama:11434"
) -> int:
    """Get the maximum safe character length for embedding a single text.
    
    This is used to truncate individual texts before embedding to avoid exceeding
    the model's context limit.
    
    Args:
        model_name: Name of the Ollama embedding model
        ollama_base_url: Base URL of Ollama API
        
    Returns:
        Maximum safe character length (tokens * 4 * 0.95 safety factor)
    """
    context_length = get_embedding_model_context_length(model_name, ollama_base_url)
    
    # Use 95% of context, 4 chars per token, rounded down
    max_chars = int(context_length * 0.95 * 4)
    max_chars = (max_chars // 100) * 100  # Round to nearest 100
    
    log.info(f"Maximum embed length for '{model_name}': {max_chars} chars ({max_chars // 4} tokens)")
    return max_chars


def calculate_optimal_max_content_size(
    llm_context_tokens: int,
    chunk_size_chars: int = 1600,
    max_docs_to_retrieve: int = 3,
    safety_margin: float = 0.75
) -> int:
    """Calculate optimal MAX_CONTENT_SIZE for RAG based on LLM's context length.
    
    MAX_CONTENT_SIZE is the maximum tokens allowed after combining:
    - Chat history
    - User input/query
    - Retrieved document context
    
    Args:
        llm_context_tokens: Maximum context length the LLM supports
        chunk_size_chars: Size of document chunks being retrieved
        max_docs_to_retrieve: Number of documents retrieved for context
        safety_margin: Fraction of context to use (0.75 = use 75%)
        
    Returns:
        Safe maximum content size in tokens
        
    Example:
        For gemma3:2b with ~8000 token context:
        - Safe budget: 8000 * 0.75 = 6000 tokens
        - Returns 6000 for MAX_CONTENT_SIZE
    """
    # Calculate safe token budget
    safe_tokens = int(llm_context_tokens * safety_margin)
    
    # Estimate context from documents (4 chars ≈ 1 token)
    estimated_doc_tokens = int((chunk_size_chars * max_docs_to_retrieve) / 4)
    
    # Leave room for retrieved context
    content_budget = safe_tokens - estimated_doc_tokens
    
    # Ensure minimum reasonable value
    content_budget = max(content_budget, 2000)
    
    log.info(
        f"Calculated MAX_CONTENT_SIZE: {content_budget} tokens "
        f"(LLM context: {llm_context_tokens}, safety margin: {int(safety_margin*100)}%)"
    )
    
    return content_budget


def calculate_optimal_docs_count(
    llm_context_tokens: int,
    chunk_size_chars: int = 1600,
    safety_margin: float = 0.6
) -> int:
    """Calculate optimal number of documents to retrieve based on LLM's context.
    
    Args:
        llm_context_tokens: Maximum context length the LLM supports
        chunk_size_chars: Size of each document chunk
        safety_margin: Fraction of context reserved for doc retrieval
        
    Returns:
        Optimal number of documents to retrieve
    """
    # Estimate tokens per chunk
    tokens_per_chunk = int(chunk_size_chars / 4)
    
    # Calculate safe token budget for documents
    safe_doc_tokens = int(llm_context_tokens * safety_margin)
    
    # Calculate how many chunks fit
    num_docs = max(1, safe_doc_tokens // tokens_per_chunk)
    
    log.info(
        f"Calculated optimal docs to retrieve: {num_docs} "
        f"(each ~{tokens_per_chunk} tokens, total budget: {safe_doc_tokens} tokens)"
    )
    
    return num_docs
