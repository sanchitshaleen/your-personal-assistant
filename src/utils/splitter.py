"""Contains a function to split text into smaller chunks."""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    DOC_CHAR_LIMIT, DOC_OVERLAP_NO, 
    ENABLE_DYNAMIC_CHUNK_SIZE, CHUNK_SIZE_SAFETY_MARGIN,
    EMB_MODEL_NAME
)

from src.core.logger import get_logger
log = get_logger(name="utils_splitter")

# Initialize dynamic chunk sizes if enabled
_CHUNK_SIZE = DOC_CHAR_LIMIT
_CHUNK_OVERLAP = DOC_OVERLAP_NO
_DYNAMIC_CHUNK_SIZE_INITIALIZED = False

def _initialize_dynamic_chunk_size():
    """Initialize dynamic chunk sizes based on embedding model's context length."""
    global _CHUNK_SIZE, _CHUNK_OVERLAP, _DYNAMIC_CHUNK_SIZE_INITIALIZED
    
    if _DYNAMIC_CHUNK_SIZE_INITIALIZED or not ENABLE_DYNAMIC_CHUNK_SIZE:
        return
    
    try:
        from src.utils.model_utils import get_dynamic_chunk_config
        
        chunk_size, chunk_overlap = get_dynamic_chunk_config(
            EMB_MODEL_NAME,
            safety_margin=CHUNK_SIZE_SAFETY_MARGIN
        )
        _CHUNK_SIZE = chunk_size
        _CHUNK_OVERLAP = chunk_overlap
        _DYNAMIC_CHUNK_SIZE_INITIALIZED = True
        
        log.info(f"Initialized dynamic chunk sizing: {_CHUNK_SIZE} chars with {_CHUNK_OVERLAP} overlap")
        
    except Exception as e:
        log.warning(f"Failed to initialize dynamic chunk size, using defaults: {e}")
        _CHUNK_SIZE = DOC_CHAR_LIMIT
        _CHUNK_OVERLAP = DOC_OVERLAP_NO
        _DYNAMIC_CHUNK_SIZE_INITIALIZED = True


def split_text(
        documents: List[Document],
        chunk_size: int = None,
        chunk_overlap: int = None
) -> tuple[bool, List[Document], str]:
    """Splits a list of Document objects into smaller chunks.

    Args:
        documents (List[Document]): List of Document objects to be split.
        chunk_size (int): The maximum size of each chunk. If None, uses config (dynamic or static).
        chunk_overlap (int): The number of characters that overlap between chunks. If None, uses config.

    Returns:
        tuple[bool, List[Document], str]: A tuple containing:
            - bool: True if the documents were split successfully, False otherwise.
            - List[Document]: A list of Document objects containing the split text.
            - str: Message indicating the result of the splitting operation.
    """
    
    # Initialize dynamic chunk sizes on first call
    _initialize_dynamic_chunk_size()
    
    # Use provided values or fall back to configured/dynamic values
    effective_chunk_size = chunk_size if chunk_size is not None else _CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else _CHUNK_OVERLAP

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=effective_chunk_size, chunk_overlap=effective_chunk_overlap
        )

        split_docs = text_splitter.split_documents(documents)

        if not split_docs:
            log.warning("No documents were split. Please check the input documents.")
            return True, [], "No documents were split. Please check the input documents."
        
        log.info(f"Successfully split {len(documents)} documents into {len(split_docs)} chunks "
                f"(size={effective_chunk_size}, overlap={effective_chunk_overlap}).")
        return True, split_docs, "Documents split successfully."

    except Exception as e:
        log.error(f"Error splitting documents: {e}")
        return False, [], f"Error splitting documents: {e}"


if __name__ == "__main__":
    # Example usage
    example_docs = [
        Document(page_content="This is a sample document. " * 10),
        Document(page_content="Another document with some text. " * 5),
        Document(page_content="Yet another document with different content. " * 3)
    ]

    status, split_documents, message = split_text(example_docs, chunk_size=100, chunk_overlap=10)

    for i, doc in enumerate(split_documents):
        print(f"Chunk {i+1}: {doc.page_content}")
        # Print first 50 characters of each chunk
        # print(f"Chunk {i+1}: {doc.page_content[:50]}...")
