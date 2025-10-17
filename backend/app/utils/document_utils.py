"""Utility functions for document operations."""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, BinaryIO
from datetime import datetime
from pathlib import Path

from ..core.logging import get_logger
from ..core.config import settings
from ..core.errors import DocumentNotFoundError, InvalidDocumentFormatError
from ..models.document import DocumentMetadata, DocumentStats

# Initialize logger
logger = get_logger("document_utils")


def generate_document_id(content: str, metadata: Dict[str, Any]) -> str:
    """Generate a unique document ID based on content and metadata.
    
    Args:
        content (str): Document content
        metadata (Dict[str, Any]): Document metadata
        
    Returns:
        str: Document ID
    """
    # Create a string to hash
    hash_str = f"{content}{json.dumps(metadata, sort_keys=True)}{datetime.now().isoformat()}"
    
    # Generate hash
    doc_id = hashlib.sha256(hash_str.encode('utf-8')).hexdigest()[:16]
    
    return f"doc-{doc_id}"


def get_document_path(document_id: str) -> str:
    """Get the path to a document file.
    
    Args:
        document_id (str): Document ID
        
    Returns:
        str: Document file path
    """
    # Get document store path from settings
    doc_store_path = Path(settings.DOCUMENT_STORE_PATH)
    
    # Create document path
    doc_path = doc_store_path / f"{document_id}.json"
    
    return str(doc_path)


def save_document_metadata(document_id: str, metadata: Dict[str, Any]) -> str:
    """Save document metadata to a file.
    
    Args:
        document_id (str): Document ID
        metadata (Dict[str, Any]): Document metadata
        
    Returns:
        str: Path to the metadata file
    """
    # Get document path
    doc_path = get_document_path(document_id)
    
    # Ensure document store directory exists
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    
    # Add timestamp to metadata
    metadata['updated_at'] = datetime.now().isoformat()
    
    # Save metadata to file
    with open(doc_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Document metadata saved: {doc_path}")
    
    return doc_path


def load_document_metadata(document_id: str) -> Dict[str, Any]:
    """Load document metadata from a file.
    
    Args:
        document_id (str): Document ID
        
    Returns:
        Dict[str, Any]: Document metadata
        
    Raises:
        DocumentNotFoundError: If the document does not exist
    """
    # Get document path
    doc_path = get_document_path(document_id)
    
    # Check if document exists
    if not os.path.exists(doc_path):
        raise DocumentNotFoundError(document_id)
    
    # Load metadata from file
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata
    except Exception as e:
        logger.error(f"Error loading document metadata: {doc_path} - {e}")
        raise


def delete_document(document_id: str) -> bool:
    """Delete a document and its metadata.
    
    Args:
        document_id (str): Document ID
        
    Returns:
        bool: True if document was deleted, False otherwise
    """
    # Get document path
    doc_path = get_document_path(document_id)
    
    # Check if document exists
    if not os.path.exists(doc_path):
        logger.warning(f"Document not found: {document_id}")
        return False
    
    # Load metadata to get file path
    try:
        metadata = load_document_metadata(document_id)
        file_path = metadata.get('file_path')
        
        # Delete the original file if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Document file deleted: {file_path}")
        
        # Delete the metadata file
        os.remove(doc_path)
        logger.info(f"Document metadata deleted: {doc_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error deleting document: {document_id} - {e}")
        return False


def list_documents() -> List[Dict[str, Any]]:
    """List all documents in the document store.
    
    Returns:
        List[Dict[str, Any]]: List of document metadata
    """
    # Get document store path from settings
    doc_store_path = Path(settings.DOCUMENT_STORE_PATH)
    
    # Check if document store directory exists
    if not os.path.exists(doc_store_path):
        return []
    
    # List all document metadata files
    documents = []
    for file_name in os.listdir(doc_store_path):
        if file_name.endswith('.json'):
            doc_id = file_name.replace('.json', '')
            try:
                metadata = load_document_metadata(doc_id)
                documents.append(metadata)
            except Exception as e:
                logger.error(f"Error loading document metadata: {doc_id} - {e}")
    
    # Sort documents by updated_at (newest first)
    documents.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    
    return documents


def get_document_stats(document_id: str) -> DocumentStats:
    """Get statistics for a document.
    
    Args:
        document_id (str): Document ID
        
    Returns:
        DocumentStats: Document statistics
        
    Raises:
        DocumentNotFoundError: If the document does not exist
    """
    # Load document metadata
    metadata = load_document_metadata(document_id)
    
    # Get document stats
    stats = metadata.get('stats', {})
    
    # Create DocumentStats object
    document_stats = DocumentStats(
        document_id=document_id,
        word_count=stats.get('word_count', 0),
        character_count=stats.get('character_count', 0),
        chunk_count=stats.get('chunk_count', 0),
        page_count=stats.get('page_count', 0),
        created_at=metadata.get('created_at'),
        updated_at=metadata.get('updated_at'),
        processing_time=stats.get('processing_time', 0),
        file_size=metadata.get('file_size', 0),
        embedding_model=metadata.get('embedding_model', settings.EMBEDDING_MODEL),
        embedding_dimension=metadata.get('embedding_dimension', settings.EMBEDDING_DIMENSION),
    )
    
    return document_stats


def update_document_stats(document_id: str, stats: Dict[str, Any]) -> DocumentStats:
    """Update statistics for a document.
    
    Args:
        document_id (str): Document ID
        stats (Dict[str, Any]): Document statistics
        
    Returns:
        DocumentStats: Updated document statistics
        
    Raises:
        DocumentNotFoundError: If the document does not exist
    """
    # Load document metadata
    metadata = load_document_metadata(document_id)
    
    # Update stats in metadata
    metadata['stats'] = stats
    metadata['updated_at'] = datetime.now().isoformat()
    
    # Save updated metadata
    save_document_metadata(document_id, metadata)
    
    # Return updated stats
    return get_document_stats(document_id)


def get_document_metadata(document_id: str) -> DocumentMetadata:
    """Get metadata for a document.
    
    Args:
        document_id (str): Document ID
        
    Returns:
        DocumentMetadata: Document metadata
        
    Raises:
        DocumentNotFoundError: If the document does not exist
    """
    # Load document metadata
    metadata = load_document_metadata(document_id)
    
    # Create DocumentMetadata object
    document_metadata = DocumentMetadata(
        document_id=document_id,
        title=metadata.get('title', ''),
        source=metadata.get('source', ''),
        source_type=metadata.get('source_type', ''),
        author=metadata.get('author', ''),
        created_at=metadata.get('created_at'),
        updated_at=metadata.get('updated_at'),
        file_path=metadata.get('file_path', ''),
        file_type=metadata.get('file_type', ''),
        file_size=metadata.get('file_size', 0),
        mime_type=metadata.get('mime_type', ''),
        language=metadata.get('language', 'en'),
        description=metadata.get('description', ''),
        tags=metadata.get('tags', []),
        metadata=metadata.get('metadata', {}),
    )
    
    return document_metadata


def update_document_metadata(document_id: str, metadata_updates: Dict[str, Any]) -> DocumentMetadata:
    """Update metadata for a document.
    
    Args:
        document_id (str): Document ID
        metadata_updates (Dict[str, Any]): Metadata updates
        
    Returns:
        DocumentMetadata: Updated document metadata
        
    Raises:
        DocumentNotFoundError: If the document does not exist
    """
    # Load document metadata
    metadata = load_document_metadata(document_id)
    
    # Update metadata
    metadata.update(metadata_updates)
    metadata['updated_at'] = datetime.now().isoformat()
    
    # Save updated metadata
    save_document_metadata(document_id, metadata)
    
    # Return updated metadata
    return get_document_metadata(document_id)


def extract_document_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from document text.
    
    Args:
        text (str): Document text
        max_keywords (int): Maximum number of keywords to extract
        
    Returns:
        List[str]: List of keywords
    """
    # Normalize text
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split into words
    words = text.split()
    
    # Remove common stop words
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'when', 'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through',
        'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on',
        'at', 'by', 'with', 'about', 'against', 'between', 'into', 'through',
        'after', 'before', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'too', 'very', 'can', 'will', 'should', 'now'
    }
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort words by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Extract top keywords
    keywords = [word for word, _ in sorted_words[:max_keywords]]
    
    return keywords


def generate_document_summary(text: str, max_length: int = 200) -> str:
    """Generate a summary for a document.
    
    Args:
        text (str): Document text
        max_length (int): Maximum summary length
        
    Returns:
        str: Document summary
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # If text is short, return it as is
    if len(text) <= max_length:
        return text
    
    # If there are only a few sentences, return the first one
    if len(sentences) <= 2:
        return sentences[0]
    
    # Otherwise, return the first sentence and truncate if necessary
    summary = sentences[0]
    
    if len(summary) > max_length:
        summary = summary[:max_length - 3] + '...'
    
    return summary


def detect_document_language(text: str) -> str:
    """Detect the language of a document.
    
    Args:
        text (str): Document text
        
    Returns:
        str: Language code (ISO 639-1)
    """
    # This is a simple implementation that only checks for a few languages
    # For production use, consider using a proper language detection library
    
    # Normalize text
    text = text.lower()
    
    # Sample of common words in different languages
    language_words = {
        'en': {'the', 'and', 'is', 'in', 'to', 'it', 'of', 'that', 'you', 'for'},
        'es': {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'ser', 'se', 'no'},
        'fr': {'le', 'la', 'de', 'et', 'est', 'en', 'un', 'que', 'qui', 'pour'},
        'de': {'der', 'die', 'das', 'und', 'ist', 'in', 'zu', 'den', 'mit', 'nicht'},
    }
    
    # Split text into words
    words = set(re.findall(r'\b\w+\b', text))
    
    # Count matches for each language
    matches = {}
    for lang, lang_words in language_words.items():
        matches[lang] = len(words.intersection(lang_words))
    
    # Return the language with the most matches
    if not matches or max(matches.values()) == 0:
        return 'unknown'
    
    return max(matches, key=matches.get)


def calculate_document_stats(text: str) -> Dict[str, Any]:
    """Calculate statistics for a document.
    
    Args:
        text (str): Document text
        
    Returns:
        Dict[str, Any]: Document statistics
    """
    # Calculate basic stats
    char_count = len(text)
    word_count = len(re.findall(r'\b\w+\b', text))
    sentence_count = len(re.split(r'(?<=[.!?])\s+', text))
    paragraph_count = len(re.split(r'\n\s*\n', text))
    
    # Calculate average word length
    words = re.findall(r'\b\w+\b', text)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Calculate average sentence length
    sentences = re.split(r'(?<=[.!?])\s+', text)
    avg_sentence_length = sum(len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences) / sentence_count if sentence_count > 0 else 0
    
    # Return stats
    return {
        'character_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'average_word_length': avg_word_length,
        'average_sentence_length': avg_sentence_length,
    }


def get_document_content_type(file_path: str) -> str:
    """Get the content type of a document.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        str: Content type
        
    Raises:
        InvalidDocumentFormatError: If the file extension is not supported
    """
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Map extension to content type
    content_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.xml': 'application/xml',
    }
    
    # Check if extension is supported
    if ext not in content_types:
        raise InvalidDocumentFormatError(ext)
    
    return content_types[ext]


def is_supported_document_type(file_path: str) -> bool:
    """Check if a document type is supported.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        bool: True if document type is supported, False otherwise
    """
    try:
        get_document_content_type(file_path)
        return True
    except InvalidDocumentFormatError:
        return False