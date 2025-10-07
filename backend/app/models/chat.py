"""Models for chat functionality."""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles for chat messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Model for a chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class SourceChunk(BaseModel):
    """Model for a source chunk referenced in an answer."""
    text: str
    document_id: str
    chunk_id: str
    metadata: Optional[Dict[str, Any]] = None


class ChatSession(BaseModel):
    """Model for a chat session."""
    id: str
    title: str
    document_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: List[ChatMessage] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chat-123",
                "title": "Discussion about Climate Change Report",
                "document_id": "doc-456",
                "created_at": "2023-06-15T10:00:00",
                "updated_at": "2023-06-15T10:30:00",
                "messages": [
                    {
                        "role": "user",
                        "content": "What are the main findings of the report?",
                        "timestamp": "2023-06-15T10:00:00"
                    },
                    {
                        "role": "assistant",
                        "content": "The main findings of the report include...",
                        "timestamp": "2023-06-15T10:00:05",
                        "metadata": {
                            "sources": [
                                {
                                    "text": "The report concludes that...",
                                    "document_id": "doc-456",
                                    "chunk_id": "chunk-789",
                                    "metadata": {"page": 5}
                                }
                            ]
                        }
                    }
                ]
            }
        }


class ChatSessionCreate(BaseModel):
    """Model for creating a new chat session."""
    title: str
    document_id: Optional[str] = None


class ChatSessionUpdate(BaseModel):
    """Model for updating a chat session."""
    title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Response model for a chat session."""
    id: str
    title: str
    document_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ChatMessageCreate(BaseModel):
    """Model for creating a new chat message."""
    content: str
    role: MessageRole = MessageRole.USER
    metadata: Optional[Dict[str, Any]] = None


class ChatMessageResponse(BaseModel):
    """Response model for a chat message."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class ChatExportFormat(str, Enum):
    """Export formats for chat history."""
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "txt"
    JSON = "json"


class ChatExportRequest(BaseModel):
    """Request model for exporting chat history."""
    session_id: str
    format: ChatExportFormat = ChatExportFormat.PDF
    include_metadata: bool = False


class ChatExportResponse(BaseModel):
    """Response model for chat export."""
    session_id: str
    format: ChatExportFormat
    filename: str
    file_size: int
    download_url: str
    expires_at: datetime