"""API endpoints for chat history and conversation management."""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    """Model for a chat message."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    doc_id: Optional[str] = None
    sources: Optional[List[dict]] = None


class ChatSession(BaseModel):
    """Model for a chat session."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    doc_ids: List[str]


class ChatHistory(BaseModel):
    """Model for chat history."""
    session_id: str
    messages: List[ChatMessage]
    has_more: bool
    total_messages: int


@router.post("/sessions", response_model=ChatSession)
async def create_chat_session(
    title: str = Body(...),
    doc_ids: List[str] = Body(...),
):
    """Create a new chat session."""
    try:
        # In a real implementation, this would create a new chat session in the database
        # For now, we'll return a mock response
        current_time = datetime.now().isoformat()
        
        return ChatSession(
            id="session-123",  # This would be a real ID in production
            title=title,
            created_at=current_time,
            updated_at=current_time,
            message_count=0,
            doc_ids=doc_ids,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=List[ChatSession])
async def list_chat_sessions(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all chat sessions."""
    try:
        # In a real implementation, this would query the database for chat sessions
        # For now, we'll return mock sessions
        current_time = datetime.now().isoformat()
        
        return [
            ChatSession(
                id="session-123",
                title="Chat about Project Proposal",
                created_at=current_time,
                updated_at=current_time,
                message_count=15,
                doc_ids=["doc-456"],
            ),
            ChatSession(
                id="session-789",
                title="Research Paper Discussion",
                created_at=current_time,
                updated_at=current_time,
                message_count=8,
                doc_ids=["doc-101", "doc-102"],
            ),
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=ChatHistory)
async def get_chat_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Get chat history for a session."""
    try:
        # In a real implementation, this would query the database for chat messages
        # For now, we'll return mock messages
        current_time = datetime.now().isoformat()
        
        messages = [
            ChatMessage(
                id="msg-1",
                role="user",
                content="What are the main findings in the research paper?",
                timestamp=current_time,
                doc_id="doc-101",
            ),
            ChatMessage(
                id="msg-2",
                role="assistant",
                content="The main findings of the research paper include three key discoveries...",
                timestamp=current_time,
                doc_id="doc-101",
                sources=[
                    {"text": "The study revealed three significant findings...", "source": "Page 12, Section 4.2"},
                    {"text": "Furthermore, the data suggests...", "source": "Page 15, Table 3"},
                ],
            ),
        ]
        
        return ChatHistory(
            session_id=session_id,
            messages=messages,
            has_more=False,
            total_messages=len(messages),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/messages", response_model=ChatMessage)
async def add_chat_message(
    session_id: str,
    role: str = Body(...),
    content: str = Body(...),
    doc_id: Optional[str] = Body(None),
    sources: Optional[List[dict]] = Body(None),
):
    """Add a message to a chat session."""
    try:
        # Validate role
        if role not in ["user", "assistant"]:
            raise HTTPException(status_code=400, detail="Role must be 'user' or 'assistant'")
        
        # In a real implementation, this would add the message to the database
        # For now, we'll return a mock response
        current_time = datetime.now().isoformat()
        
        return ChatMessage(
            id="msg-new",  # This would be a real ID in production
            role=role,
            content=content,
            timestamp=current_time,
            doc_id=doc_id,
            sources=sources,
        )
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    try:
        # In a real implementation, this would delete the session from the database
        # For now, we'll return a mock response
        return {"message": f"Chat session {session_id} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/export")
async def export_chat_history(
    session_id: str,
    format: str = Body(...),  # "pdf" or "docx"
):
    """Export chat history as PDF or DOCX."""
    try:
        # Validate format
        if format not in ["pdf", "docx"]:
            raise HTTPException(status_code=400, detail="Format must be 'pdf' or 'docx'")
        
        # In a real implementation, this would generate the document and return a download URL
        # For now, we'll return a mock response
        return {
            "download_url": f"/api/downloads/chat-export-{session_id}.{format}",
            "expires_at": (datetime.now().timestamp() + 3600) * 1000,  # 1 hour from now, in milliseconds
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))