"""Error handling for the SmartDocQ application."""

from fastapi import HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any


class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str
    error_code: str
    status_code: int
    metadata: Optional[Dict[str, Any]] = None


class SmartDocQException(Exception):
    """Base exception for SmartDocQ application."""
    def __init__(
        self, 
        detail: str, 
        error_code: str, 
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.detail = detail
        self.error_code = error_code
        self.status_code = status_code
        self.metadata = metadata
        super().__init__(self.detail)


class DocumentProcessingError(SmartDocQException):
    """Exception raised when document processing fails."""
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="DOCUMENT_PROCESSING_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            metadata=metadata
        )


class DocumentNotFoundError(SmartDocQException):
    """Exception raised when a document is not found."""
    def __init__(self, document_id: str):
        super().__init__(
            detail=f"Document with ID {document_id} not found",
            error_code="DOCUMENT_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            metadata={"document_id": document_id}
        )


class ValidationError(SmartDocQException):
    """Exception raised when validation fails."""
    def __init__(self, detail: str, field: Optional[str] = None):
        metadata = {"field": field} if field else None
        super().__init__(
            detail=detail,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            metadata=metadata
        )


class InvalidDocumentFormatError(SmartDocQException):
    """Exception raised when document format is invalid."""
    def __init__(self, file_extension: str):
        super().__init__(
            detail=f"Invalid document format: {file_extension}",
            error_code="INVALID_DOCUMENT_FORMAT",
            status_code=status.HTTP_400_BAD_REQUEST,
            metadata={"file_extension": file_extension}
        )


class DocumentTooLargeError(SmartDocQException):
    """Exception raised when document size exceeds the limit."""
    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            detail=f"Document size ({file_size} bytes) exceeds the maximum allowed size ({max_size} bytes)",
            error_code="DOCUMENT_TOO_LARGE",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            metadata={"file_size": file_size, "max_size": max_size}
        )


class VectorStoreError(SmartDocQException):
    """Exception raised when vector store operations fail."""
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="VECTOR_STORE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            metadata=metadata
        )


class EmbeddingGenerationError(SmartDocQException):
    """Exception raised when embedding generation fails."""
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="EMBEDDING_GENERATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            metadata=metadata
        )


class QuestionAnsweringError(SmartDocQException):
    """Exception raised when question answering fails."""
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="QUESTION_ANSWERING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            metadata=metadata
        )


class OCRProcessingError(SmartDocQException):
    """Exception raised when OCR processing fails."""
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="OCR_PROCESSING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            metadata=metadata
        )


class SpeechProcessingError(SmartDocQException):
    """Exception raised when speech processing fails."""
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="SPEECH_PROCESSING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            metadata=metadata
        )


class ChatSessionNotFoundError(SmartDocQException):
    """Exception raised when a chat session is not found."""
    def __init__(self, session_id: str):
        super().__init__(
            detail=f"Chat session with ID {session_id} not found",
            error_code="CHAT_SESSION_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            metadata={"session_id": session_id}
        )


class InvalidURLError(SmartDocQException):
    """Exception raised when a URL is invalid."""
    def __init__(self, url: str):
        super().__init__(
            detail=f"Invalid URL: {url}",
            error_code="INVALID_URL",
            status_code=status.HTTP_400_BAD_REQUEST,
            metadata={"url": url}
        )


class WebPageProcessingError(SmartDocQException):
    """Exception raised when web page processing fails."""
    def __init__(self, url: str, detail: str):
        super().__init__(
            detail=f"Failed to process web page at {url}: {detail}",
            error_code="WEB_PAGE_PROCESSING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            metadata={"url": url}
        )


class TranslationError(SmartDocQException):
    """Exception raised when translation fails."""
    def __init__(self, detail: str, source_language: Optional[str] = None, target_language: Optional[str] = None):
        metadata = {}
        if source_language:
            metadata["source_language"] = source_language
        if target_language:
            metadata["target_language"] = target_language
            
        super().__init__(
            detail=detail,
            error_code="TRANSLATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            metadata=metadata
        )


class FeedbackValidationError(SmartDocQException):
    """Exception raised when feedback validation fails."""
    def __init__(self, detail: str):
        super().__init__(
            detail=detail,
            error_code="FEEDBACK_VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST
        )


class QuestionGenerationError(SmartDocQException):
    """Exception raised when question generation fails."""
    def __init__(self, detail: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="QUESTION_GENERATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            metadata=metadata
        )


def http_exception_handler(exc: SmartDocQException):
    """Convert SmartDocQException to FastAPI HTTPException.
    
    Args:
        exc (SmartDocQException): Exception to convert
        
    Returns:
        HTTPException: FastAPI HTTP exception
    """
    return HTTPException(
        status_code=exc.status_code,
        detail=ErrorResponse(
            detail=exc.detail,
            error_code=exc.error_code,
            status_code=exc.status_code,
            metadata=exc.metadata
        ).dict()
    )