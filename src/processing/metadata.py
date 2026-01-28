"""
Metadata Extractor Module
- Extracts document metadata (creation date, author, etc.) from various file types
- Currently supports: PDF, DOCX
"""

import fitz  # PyMuPDF for PDF
from src.core.logger import get_logger
from datetime import datetime
import pytz

log = get_logger(name="METADATA_EXTRACTOR")
CST = pytz.timezone('America/Chicago')


def extract_pdf_creation_date(file_path: str) -> str:
    """Extracts the creation date from PDF metadata.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Creation date in format "YYYY-MM-DD HH:MM:SS" or None if not found
    """
    try:
        doc = fitz.open(file_path)
        metadata = doc.metadata
        
        if metadata and 'creationDate' in metadata:
            creation_date = metadata['creationDate']
            # PDF dates are in format: D:YYYYMMDDHHmmSS
            if creation_date:
                try:
                    # Remove 'D:' prefix and parse
                    date_str = creation_date
                    if date_str.startswith('D:'):
                        date_str = date_str[2:]
                    
                    # Parse the date (at minimum YYYYMMDD, can be YYYYMMDDHHmmSS)
                    if len(date_str) >= 8:
                        parsed_date = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S" if len(date_str) >= 14 else "%Y%m%d000000")
                        # Convert to CST
                        utc_date = pytz.utc.localize(parsed_date)
                        cst_date = utc_date.astimezone(CST)
                        return cst_date.strftime("%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    log.warning(f"Could not parse PDF creation date '{creation_date}': {e}")
        
        doc.close()
        return None
        
    except Exception as e:
        log.warning(f"Could not extract PDF metadata from {file_path}: {e}")
        return None


def extract_docx_creation_date(file_path: str) -> str:
    """Extracts the creation date from DOCX metadata.
    
    Args:
        file_path (str): Path to the DOCX file
        
    Returns:
        str: Creation date in format "YYYY-MM-DD HH:MM:SS" or None if not found
    """
    try:
        from docx import Document
        from docx.oxml import parse_xml
        
        doc = Document(file_path)
        core_props = doc.core_properties
        
        if core_props.created:
            created_date = core_props.created
            # Convert to CST
            if created_date.tzinfo is None:
                created_date = pytz.utc.localize(created_date)
            else:
                created_date = created_date.astimezone(CST)
            return created_date.strftime("%Y-%m-%d %H:%M:%S")
        
        return None
        
    except Exception as e:
        log.warning(f"Could not extract DOCX metadata from {file_path}: {e}")
        return None


def extract_source_creation_date(file_path: str, file_type: str = None) -> str:
    """Extracts the source document creation date based on file type.
    
    Args:
        file_path (str): Path to the file
        file_type (str): MIME type (e.g., application/pdf, application/vnd.openxmlformats-officedocument.wordprocessingml.document)
        
    Returns:
        str: Creation date in format "YYYY-MM-DD HH:MM:SS" or None if not found
    """
    if not file_type:
        # Infer from file extension
        if file_path.lower().endswith('.pdf'):
            file_type = 'application/pdf'
        elif file_path.lower().endswith('.docx'):
            file_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    
    if file_type and 'pdf' in file_type.lower():
        return extract_pdf_creation_date(file_path)
    elif file_type and ('wordprocessingml' in file_type.lower() or 'msword' in file_type.lower()):
        return extract_docx_creation_date(file_path)
    
    return None
