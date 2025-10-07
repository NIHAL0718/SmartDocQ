"""Data models for question answering functionality."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Model for a chat message in the conversation history."""
    role: str  # "user" or "assistant"
    content: str


class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str
    doc_id: str
    language: Optional[str] = "english"
    max_sources: Optional[int] = 5
    chat_history: Optional[List[ChatMessage]] = None
    use_semantic_search: Optional[bool] = True
    include_sources: Optional[bool] = True


class SourceChunk(BaseModel):
    """Model for a source chunk used to answer a question."""
    text: str
    source: str  # e.g., "Page 5, Paragraph 2" or "Section 3.1"
    relevance_score: Optional[float] = None
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None


class AnswerResponse(BaseModel):
    """Response model for question answering."""
    answer: str
    sources: Optional[List[SourceChunk]] = None
    doc_id: str
    question: str
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None  # in seconds
    audio_answer_url: Optional[str] = None  # URL to audio version of the answer
    follow_up_questions: Optional[List[str]] = None


class QuestionHistory(BaseModel):
    """Model for question history."""
    id: str
    question: str
    answer: str
    doc_id: str
    timestamp: str
    sources: Optional[List[SourceChunk]] = None
    feedback: Optional[Dict[str, Any]] = None


class VoiceQuestionRequest(BaseModel):
    """Request model for voice question."""
    doc_id: str
    language: Optional[str] = "english"
    audio_format: Optional[str] = "wav"  # wav, mp3, etc.
    max_sources: Optional[int] = 5
    chat_history: Optional[List[ChatMessage]] = None
    return_audio: Optional[bool] = True


class TranslationRequest(BaseModel):
    """Model for a translation request."""
    text: str
    target_language: str
    source_language: Optional[str] = None


class TextToTextTranslationRequest(BaseModel):
    """Model for a text-to-text translation request."""
    text: str
    target_language: str
    source_language: Optional[str] = None
    use_text_to_text: bool = True


class TranslationResponse(BaseModel):
    """Model for a translation response."""
    original_text: str
    translated_text: str
    target_language: str
    source_language: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())