"""Utility functions for file operations."""

import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, BinaryIO
from datetime import datetime

from fastapi import UploadFile

from ..core.config import settings
from ..core.logging import get_logger
from ..core.errors import DocumentTooLargeError, InvalidDocumentFormatError

# Initialize logger
logger = get_logger("file_utils")

# Define allowed file extensions
ALLOWED_DOCUMENT_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

# Define document storage directory
DOCUMENT_DIR = Path(settings.DOCUMENT_STORE_PATH)
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)


async def save_uploaded_file(file: UploadFile, directory: Optional[str] = None) -> Tuple[str, str, int]:
    """Save an uploaded file to the specified directory.
    
    Args:
        file (UploadFile): The uploaded file
        directory (Optional[str]): Directory to save the file (relative to DOCUMENT_STORE_PATH)
        
    Returns:
        Tuple[str, str, int]: Tuple of (file_id, file_path, file_size)
        
    Raises:
        DocumentTooLargeError: If file size exceeds the maximum allowed size
        InvalidDocumentFormatError: If file extension is not allowed
    """
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_DOCUMENT_EXTENSIONS and file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise InvalidDocumentFormatError(file_ext)
    
    # Generate a unique file ID
    file_id = f"file-{uuid.uuid4()}"
    
    # Determine the save directory
    if directory:
        save_dir = DOCUMENT_DIR / directory
    else:
        save_dir = DOCUMENT_DIR
    
    # Create the directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{file_id}_{timestamp}{file_ext}"
    file_path = save_dir / filename
    
    # Check file size
    file_size = 0
    max_size_bytes = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024
    
    # Save the file
    try:
        with open(file_path, "wb") as f:
            # Read and write the file in chunks
            chunk_size = 1024 * 1024  # 1 MB
            chunk = await file.read(chunk_size)
            while chunk:
                file_size += len(chunk)
                
                # Check if file size exceeds the maximum allowed size
                if file_size > max_size_bytes:
                    # Remove the partially saved file
                    os.remove(file_path)
                    raise DocumentTooLargeError(file_size, max_size_bytes)
                
                f.write(chunk)
                chunk = await file.read(chunk_size)
    except Exception as e:
        # Clean up if an error occurs
        if file_path.exists():
            os.remove(file_path)
        logger.error(f"Error saving uploaded file: {e}")
        raise
    
    logger.info(f"File saved: {file_path} ({file_size} bytes)")
    
    # Return the file ID, path, and size
    return file_id, str(file_path), file_size


def get_file_metadata(file_path: str) -> dict:
    """Get metadata for a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        dict: File metadata
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        return {"error": "File not found"}
    
    # Get file stats
    stats = path.stat()
    
    # Get file extension
    file_ext = path.suffix.lower()
    
    # Determine file type
    if file_ext in ALLOWED_DOCUMENT_EXTENSIONS:
        file_type = "document"
        mime_type = ALLOWED_DOCUMENT_EXTENSIONS[file_ext]
    elif file_ext in ALLOWED_IMAGE_EXTENSIONS:
        file_type = "image"
        mime_type = ALLOWED_IMAGE_EXTENSIONS[file_ext]
    else:
        file_type = "unknown"
        mime_type = "application/octet-stream"
    
    # Return metadata
    return {
        "filename": path.name,
        "file_path": str(path),
        "file_size": stats.st_size,
        "file_type": file_type,
        "mime_type": mime_type,
        "extension": file_ext,
        "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
    }


def delete_file(file_path: str) -> bool:
    """Delete a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if file was deleted, False otherwise
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        logger.warning(f"File not found: {file_path}")
        return False
    
    # Delete the file
    try:
        os.remove(path)
        logger.info(f"File deleted: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file: {file_path} - {e}")
        return False


def list_files(directory: str, extensions: Optional[List[str]] = None) -> List[dict]:
    """List files in a directory.
    
    Args:
        directory (str): Directory to list files from
        extensions (Optional[List[str]]): List of file extensions to filter by
        
    Returns:
        List[dict]: List of file metadata
    """
    dir_path = Path(directory)
    
    # Check if directory exists
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    # List files
    files = []
    for item in dir_path.iterdir():
        if item.is_file():
            # Filter by extension if specified
            if extensions and item.suffix.lower() not in extensions:
                continue
            
            # Get file metadata
            metadata = get_file_metadata(str(item))
            files.append(metadata)
    
    # Sort files by modified time (newest first)
    files.sort(key=lambda x: x["modified_at"], reverse=True)
    
    return files


def create_directory(directory: str) -> bool:
    """Create a directory.
    
    Args:
        directory (str): Directory to create
        
    Returns:
        bool: True if directory was created, False otherwise
    """
    dir_path = Path(directory)
    
    # Create the directory
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created: {directory}")
        return True
    except Exception as e:
        logger.error(f"Error creating directory: {directory} - {e}")
        return False


def get_file_extension(filename: str) -> str:
    """Get the extension of a file.
    
    Args:
        filename (str): Filename
        
    Returns:
        str: File extension (with dot)
    """
    return os.path.splitext(filename)[1].lower()


def is_valid_document_extension(extension: str) -> bool:
    """Check if a file extension is valid for documents.
    
    Args:
        extension (str): File extension (with dot)
        
    Returns:
        bool: True if extension is valid, False otherwise
    """
    return extension in ALLOWED_DOCUMENT_EXTENSIONS


def is_valid_image_extension(extension: str) -> bool:
    """Check if a file extension is valid for images.
    
    Args:
        extension (str): File extension (with dot)
        
    Returns:
        bool: True if extension is valid, False otherwise
    """
    return extension in ALLOWED_IMAGE_EXTENSIONS


def get_mime_type(extension: str) -> str:
    """Get the MIME type for a file extension.
    
    Args:
        extension (str): File extension (with dot)
        
    Returns:
        str: MIME type
    """
    if extension in ALLOWED_DOCUMENT_EXTENSIONS:
        return ALLOWED_DOCUMENT_EXTENSIONS[extension]
    elif extension in ALLOWED_IMAGE_EXTENSIONS:
        return ALLOWED_IMAGE_EXTENSIONS[extension]
    else:
        return "application/octet-stream"