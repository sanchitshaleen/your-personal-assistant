"""Module dealing specifically with loading files into Document objects.
Contains the `load_file` function to load text, PDF, and markdown files.
Planning to add more file types in the future.

## For testing:
- Run this file from `server` folder as:
- `python -m llm_system.utils.loader`
"""

import os
import fitz  # PyMuPDF for more efficient PDF handling
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from src.core.logger import get_logger
log = get_logger(name="doc_loader")


def load_file(user_id: str, file_path: str) -> tuple[bool, List[Document], str]:
    """Load a file and return its content as a list of Document objects. Usually one document per page.

    Args:
        user_id (str): The ID of the user who is loading the file.
        file_path (str): The absolute path to the file to be loaded.

    Returns:
        tuple[bool, List[Document], str]: A tuple containing:
            - bool: True if the file was loaded successfully, False otherwise.
            - List[Document]: A list of Document objects containing the file's content.
            - str: Message indicating the result of the loading operation.
    """

    # Planning to add many types in future, but for now, only txt and pdf are supported:
    file_extension = file_path.split('.')[-1].lower()

    if file_extension not in ['txt', 'pdf', "md"]:
        log.error(f"Unsupported file type: {file_extension}.")
        return False, [], f"Unsupported file type: {file_extension}. Supported types are: txt, pdf."

    try:
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            file_content = loader.load()

        elif file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
            file_content = loader.load()

        elif file_path.endswith('.pdf'):
            # Use memory-efficient PDF loading with PyMuPDF (fitz)
            # Process page by page instead of loading entire PDF at once
            file_content = _load_pdf_memory_efficient(file_path, user_id)
            if file_content is None:
                return False, [], f"Failed to load PDF: {file_path}"

        # Add user metadata to each doc:
        for doc in file_content:
            doc.metadata['user_id'] = user_id
            # Since i am exposing the retrieved docs to UI
            # Hide full server file path if its there:
            if 'file_path' in doc.metadata:
                doc.metadata['file_path'] = os.path.basename(doc.metadata['file_path'])

            if 'source' in doc.metadata:
                # If it is not local file, keep source as is:
                if "www." in doc.metadata['source'] or "http" in doc.metadata['source']:
                    continue
                # If it is local file, keep only the file name:
                else:
                    doc.metadata['source'] = os.path.basename(doc.metadata['source'])
                    # Set filename to match source (for backward compatibility with code expecting 'filename')
                    doc.metadata['filename'] = doc.metadata['source']

        if not file_content:
            log.error(f"No content found in the file: {file_path}")
            return True, [], f"No content found in the file: {file_path}"

        log.info(f"Loaded {len(file_content)} documents from {file_path} for user {user_id}.")
        return True, file_content, f"Loaded {len(file_content)} documents."

    except Exception as e:
        log.error(f"Error loading file {file_path}: {repr(e)}")
        return False, [], f"Error loading file: {str(e)}"


def _load_pdf_memory_efficient(file_path: str, user_id: str) -> List[Document]:
    """Load PDF file with memory efficiency by processing pages individually.
    
    Args:
        file_path: Path to PDF file
        user_id: User ID for metadata
        
    Returns:
        List of Document objects, one per page
    """
    documents = []
    
    try:
        # Open PDF with fitz (PyMuPDF)
        pdf_document = fitz.open(file_path)
        total_pages = len(pdf_document)
        log.info(f"Loading PDF with {total_pages} pages: {file_path}")
        
        # Process each page individually to minimize memory usage
        for page_num in range(total_pages):
            try:
                # Get the page
                page = pdf_document[page_num]
                
                # Extract text from page
                page_text = page.get_text()
                
                if page_text.strip():  # Only add if page has text
                    # Create a Document object for this page
                    doc = Document(
                        page_content=page_text,
                        metadata={
                            'source': file_path,
                            'file_path': file_path,
                            'page': page_num,
                            'user_id': user_id
                        }
                    )
                    documents.append(doc)
                    
            except Exception as e:
                log.warning(f"Error processing page {page_num} of {file_path}: {repr(e)}")
                continue
        
        # Close the PDF document to free memory
        pdf_document.close()
        
        log.info(f"Successfully loaded {len(documents)} pages from PDF: {file_path}")
        return documents
        
    except Exception as e:
        log.error(f"Error loading PDF {file_path}: {repr(e)}")
        return None


if __name__ == "__main__":
    # Example usage
    try:
        status, docs, message = load_file(
            user_id="test_user",
            file_path="../../../GenAI/Data/attention_is_all_you_need_1706.03762v7.pdf"
            # file_path="../../../GenAI/Data/speech.txt"
            # file_path="../../../GenAI/Data/speech.md"
        )

        print(status)
        print(message)
        print(len(docs))

        for ind, doc in enumerate(docs[:3]):
            print("\n")
            print(repr(doc))

    except Exception as e:
        print(f"Error loading file: {e}")
