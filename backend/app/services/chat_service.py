"""Service for handling chat sessions and messages."""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

from ..core.config import settings
from ..core.logging import get_logger
from ..core.errors import ChatSessionNotFoundError
from ..models.chat import (
    ChatSession,
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionResponse,
    ChatMessage,
    ChatMessageCreate,
    ChatMessageResponse,
    ChatExportFormat,
    ChatExportRequest,
    ChatExportResponse,
    MessageRole
)

# Initialize logger
logger = get_logger("chat_service")

# Define paths for storing chat data
CHAT_DIR = Path(settings.DOCUMENT_STORE_PATH) / "chats"
CHAT_DIR.mkdir(parents=True, exist_ok=True)


class ChatService:
    """Service for handling chat sessions and messages."""
    
    @staticmethod
    def create_session(session_data: ChatSessionCreate) -> ChatSessionResponse:
        """Create a new chat session.
        
        Args:
            session_data (ChatSessionCreate): Session creation data
            
        Returns:
            ChatSessionResponse: Created session response
        """
        # Generate session ID
        session_id = f"chat-{uuid.uuid4()}"
        
        # Create timestamp
        now = datetime.now()
        
        # Create session
        session = ChatSession(
            id=session_id,
            title=session_data.title,
            document_id=session_data.document_id,
            created_at=now,
            updated_at=now,
            messages=[]
        )
        
        # Save session
        ChatService._save_session(session)
        
        # Log session creation
        logger.info(f"Chat session created: {session_id}")
        
        # Return session response
        return ChatSessionResponse(
            id=session.id,
            title=session.title,
            document_id=session.document_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=0
        )
    
    @staticmethod
    def get_session(session_id: str) -> ChatSession:
        """Get a chat session by ID.
        
        Args:
            session_id (str): Session ID
            
        Returns:
            ChatSession: Chat session
            
        Raises:
            ChatSessionNotFoundError: If session not found
        """
        # Get session file path
        session_file = CHAT_DIR / f"{session_id}.json"
        
        # Check if session exists
        if not session_file.exists():
            raise ChatSessionNotFoundError(session_id)
        
        # Load session from file
        try:
            with open(session_file, "r") as f:
                session_data = json.load(f)
            
            # Convert to ChatSession model
            session = ChatSession(**session_data)
            
            return session
        except Exception as e:
            logger.error(f"Error loading chat session {session_id}: {e}")
            raise ChatSessionNotFoundError(session_id)
    
    @staticmethod
    def list_sessions() -> List[ChatSessionResponse]:
        """List all chat sessions.
        
        Returns:
            List[ChatSessionResponse]: List of chat session responses
        """
        # Get all session files
        session_files = list(CHAT_DIR.glob("*.json"))
        
        # Load sessions from files
        sessions = []
        for file in session_files:
            try:
                with open(file, "r") as f:
                    session_data = json.load(f)
                
                # Convert to ChatSession model
                session = ChatSession(**session_data)
                
                # Create session response
                session_response = ChatSessionResponse(
                    id=session.id,
                    title=session.title,
                    document_id=session.document_id,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    message_count=len(session.messages)
                )
                
                sessions.append(session_response)
            except Exception as e:
                logger.error(f"Error loading chat session from {file}: {e}")
        
        # Sort sessions by updated_at (newest first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        
        return sessions
    
    @staticmethod
    def update_session(session_id: str, update_data: ChatSessionUpdate) -> ChatSessionResponse:
        """Update a chat session.
        
        Args:
            session_id (str): Session ID
            update_data (ChatSessionUpdate): Session update data
            
        Returns:
            ChatSessionResponse: Updated session response
            
        Raises:
            ChatSessionNotFoundError: If session not found
        """
        # Get session
        session = ChatService.get_session(session_id)
        
        # Update session
        if update_data.title is not None:
            session.title = update_data.title
        
        # Update timestamp
        session.updated_at = datetime.now()
        
        # Save session
        ChatService._save_session(session)
        
        # Log session update
        logger.info(f"Chat session updated: {session_id}")
        
        # Return session response
        return ChatSessionResponse(
            id=session.id,
            title=session.title,
            document_id=session.document_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=len(session.messages)
        )
    
    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Delete a chat session.
        
        Args:
            session_id (str): Session ID
            
        Returns:
            bool: True if session was deleted
            
        Raises:
            ChatSessionNotFoundError: If session not found
        """
        # Get session file path
        session_file = CHAT_DIR / f"{session_id}.json"
        
        # Check if session exists
        if not session_file.exists():
            raise ChatSessionNotFoundError(session_id)
        
        # Delete session file
        try:
            os.remove(session_file)
            logger.info(f"Chat session deleted: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chat session {session_id}: {e}")
            return False
    
    @staticmethod
    def add_message(session_id: str, message_data: ChatMessageCreate) -> ChatMessageResponse:
        """Add a message to a chat session.
        
        Args:
            session_id (str): Session ID
            message_data (ChatMessageCreate): Message creation data
            
        Returns:
            ChatMessageResponse: Created message response
            
        Raises:
            ChatSessionNotFoundError: If session not found
        """
        # Get session
        session = ChatService.get_session(session_id)
        
        # Generate message ID
        message_id = f"msg-{uuid.uuid4()}"
        
        # Create message
        message = ChatMessage(
            role=message_data.role,
            content=message_data.content,
            timestamp=datetime.now(),
            metadata=message_data.metadata
        )
        
        # Add message to session
        session.messages.append(message)
        
        # Update session timestamp
        session.updated_at = datetime.now()
        
        # Save session
        ChatService._save_session(session)
        
        # Log message addition
        logger.info(f"Message added to chat session {session_id}")
        
        # Return message response
        return ChatMessageResponse(
            id=message_id,
            role=message.role,
            content=message.content,
            timestamp=message.timestamp,
            metadata=message.metadata
        )
    
    @staticmethod
    def get_messages(session_id: str) -> List[ChatMessageResponse]:
        """Get all messages in a chat session.
        
        Args:
            session_id (str): Session ID
            
        Returns:
            List[ChatMessageResponse]: List of chat message responses
            
        Raises:
            ChatSessionNotFoundError: If session not found
        """
        # Get session
        session = ChatService.get_session(session_id)
        
        # Convert messages to response models
        message_responses = []
        for i, message in enumerate(session.messages):
            message_id = f"msg-{i+1}"
            message_response = ChatMessageResponse(
                id=message_id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
                metadata=message.metadata
            )
            message_responses.append(message_response)
        
        return message_responses
    
    @staticmethod
    def export_chat(export_request: ChatExportRequest) -> ChatExportResponse:
        """Export a chat session to a file.
        
        Args:
            export_request (ChatExportRequest): Export request data
            
        Returns:
            ChatExportResponse: Export response
            
        Raises:
            ChatSessionNotFoundError: If session not found
        """
        # Get session
        session = ChatService.get_session(export_request.session_id)
        
        # Generate export file name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"chat_export_{session.id}_{timestamp}.{export_request.format}"
        
        # Create export directory if it doesn't exist
        export_dir = Path(settings.DOCUMENT_STORE_PATH) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export file path
        export_file = export_dir / filename
        
        # Export chat based on format
        if export_request.format == ChatExportFormat.JSON:
            # Export as JSON
            export_data = {
                "session_id": session.id,
                "title": session.title,
                "document_id": session.document_id,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "messages": [
                    {
                        "role": message.role,
                        "content": message.content,
                        "timestamp": message.timestamp.isoformat(),
                        "metadata": message.metadata if export_request.include_metadata else None
                    }
                    for message in session.messages
                ]
            }
            
            with open(export_file, "w") as f:
                json.dump(export_data, f, indent=2)
        
        elif export_request.format == ChatExportFormat.TEXT:
            # Export as plain text
            with open(export_file, "w") as f:
                f.write(f"Chat Session: {session.title}\n")
                f.write(f"Session ID: {session.id}\n")
                f.write(f"Document ID: {session.document_id}\n")
                f.write(f"Created: {session.created_at}\n")
                f.write(f"Updated: {session.updated_at}\n\n")
                
                f.write("Messages:\n\n")
                for message in session.messages:
                    f.write(f"{message.role.capitalize()}: {message.content}\n")
                    f.write(f"Time: {message.timestamp}\n\n")
        
        else:
            # For PDF and DOCX formats, we would use libraries like reportlab or python-docx
            # This is a placeholder implementation
            with open(export_file, "w") as f:
                f.write(f"Chat export in {export_request.format} format\n")
                f.write(f"This is a placeholder for {export_request.format} export")
        
        # Get file size
        file_size = os.path.getsize(export_file)
        
        # Generate download URL (in a real app, this would be a proper URL)
        download_url = f"/api/chats/{session.id}/exports/{filename}"
        
        # Set expiration time (24 hours from now)
        expires_at = datetime.now().replace(hour=23, minute=59, second=59)
        
        # Log export
        logger.info(f"Chat session {session.id} exported to {filename}")
        
        # Return export response
        return ChatExportResponse(
            session_id=session.id,
            format=export_request.format,
            filename=filename,
            file_size=file_size,
            download_url=download_url,
            expires_at=expires_at
        )
    
    @staticmethod
    def _save_session(session: ChatSession) -> None:
        """Save a chat session to file.
        
        Args:
            session (ChatSession): Chat session to save
        """
        # Convert session to dict
        session_dict = session.dict()
        
        # Save session to file
        session_file = CHAT_DIR / f"{session.id}.json"
        with open(session_file, "w") as f:
            json.dump(session_dict, f, indent=2)