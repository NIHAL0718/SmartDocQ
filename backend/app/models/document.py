"""Data models for document processing."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """Model for a document chunk."""
    id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class DocumentMetadata(BaseModel):
    """Model for document metadata."""
    id: str
    title: str
    upload_date: str
    file_type: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    language: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    source_url: Optional[str] = None


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    id: str
    title: str
    status: str  # "processing", "completed", "failed"
    message: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    page_count: Optional[int] = 0
    chunk_count: Optional[int] = 0
    word_count: Optional[int] = 0


class DocumentProcessingResult(BaseModel):
    """Model for document processing result."""
    doc_id: str
    title: str
    text: str
    chunks: List[DocumentChunk]
    metadata: DocumentMetadata
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    processing_time: float  # in seconds


class DocumentSummary(BaseModel):
    """Model for document summary."""
    id: str
    title: str
    summary: str
    key_points: List[str]
    entities: Dict[str, List[str]]  # e.g., {"people": ["John Doe"], "organizations": ["Acme Inc."]}
    topics: List[str]


class DocumentStats(BaseModel):
    """Model for document statistics."""
    doc_id: str
    word_count: int
    page_count: int
    chunk_count: int
    question_count: int
    average_chunk_size: int
    languages: List[str]
    creation_date: str
    last_accessed: str