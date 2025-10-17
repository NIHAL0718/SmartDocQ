"""Service for document processing and management."""

import os
import time
from typing import List, Dict, Any, Optional, BinaryIO
from datetime import datetime
import uuid

from ..core.logging import get_logger
from ..core.config import settings
from ..core.errors import (
    DocumentNotFoundError, 
    InvalidDocumentFormatError,
    DocumentSizeExceededError,
    VectorStoreError,
    EmbeddingGenerationError
)
from ..models.document import (
    DocumentMetadata,
    DocumentResponse,
    DocumentChunk,
    DocumentStats,
    DocumentSummary
)
from ..utils.document_utils import (
    generate_document_id,
    save_document_metadata,
    load_document_metadata,
    delete_document,
    list_documents,
    get_document_stats,
    update_document_stats,
    get_document_metadata,
    update_document_metadata,
    extract_document_keywords,
    generate_document_summary,
    detect_document_language,
    calculate_document_stats,
    get_document_content_type,
    is_supported_document_type
)
from ..utils.file_utils import (
    save_uploaded_file,
    get_file_metadata,
    get_file_extension,
    is_valid_document_extension
)
from ..utils.text_utils import (
    normalize_text,
    split_text_into_chunks
)

# Initialize logger
logger = get_logger("document_service")


class DocumentService:
    """Service for document processing and management."""
    
    def __init__(self, vector_store=None, embedding_service=None):
        """Initialize the document service.
        
        Args:
            vector_store: Vector store for document embeddings
            embedding_service: Service for generating embeddings
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def process_document(self, 
                         file: BinaryIO, 
                         filename: str, 
                         title: Optional[str] = None,
                         description: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> DocumentResponse:
        """Process a document file.
        
        Args:
            file: File object
            filename: Original filename
            title: Document title (optional)
            description: Document description (optional)
            tags: Document tags (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            DocumentResponse: Document response
            
        Raises:
            InvalidDocumentFormatError: If the document format is not supported
            DocumentSizeExceededError: If the document size exceeds the maximum allowed
        """
        start_time = time.time()
        
        # Check if file extension is supported
        if not is_valid_document_extension(filename):
            raise InvalidDocumentFormatError(f"Unsupported file extension: {filename}")
        
        # Save uploaded file
        file_path = await save_uploaded_file(file, filename, settings.DOCUMENT_STORE_PATH)
        
        # Get file metadata
        file_metadata = get_file_metadata(file_path)
        
        # Check file size
        if file_metadata['size'] > settings.MAX_DOCUMENT_SIZE:
            # Delete the file
            os.remove(file_path)
            raise DocumentSizeExceededError(f"Document size exceeds maximum allowed: {file_metadata['size']} > {settings.MAX_DOCUMENT_SIZE}")
        
        # Extract text from document
        text = await self._extract_text_from_document(file_path, file_metadata['extension'])
        
        # Generate document ID
        doc_id = generate_document_id(text, file_metadata)
        
        # Use provided title or filename as title
        doc_title = title or os.path.splitext(os.path.basename(filename))[0]
        
        # Detect language
        language = detect_document_language(text)
        
        # Calculate document statistics
        stats = calculate_document_stats(text)
        
        # Split text into chunks
        chunks = await self._split_document_into_chunks(doc_id, text)
        
        # Generate embeddings for chunks
        await self._generate_embeddings_for_chunks(chunks)
        
        # Store chunks in vector store
        await self._store_chunks_in_vector_store(chunks)
        
        # Generate document summary
        summary = generate_document_summary(text)
        
        # Extract keywords
        keywords = extract_document_keywords(text)
        
        # Create document metadata
        doc_metadata = {
            'document_id': doc_id,
            'title': doc_title,
            'source': filename,
            'source_type': 'file',
            'file_path': file_path,
            'file_type': file_metadata['extension'],
            'file_size': file_metadata['size'],
            'mime_type': file_metadata['mime_type'],
            'language': language,
            'description': description or summary,
            'tags': tags or keywords,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'stats': {
                **stats,
                'chunk_count': len(chunks),
                'processing_time': time.time() - start_time,
                'embedding_model': settings.EMBEDDING_MODEL,
                'embedding_dimension': settings.EMBEDDING_DIMENSION,
            },
            'metadata': metadata or {},
        }
        
        # Save document metadata
        save_document_metadata(doc_id, doc_metadata)
        
        # Create document response
        response = DocumentResponse(
            document_id=doc_id,
            title=doc_title,
            source=filename,
            source_type='file',
            author=metadata.get('author') if metadata else None,
            created_at=doc_metadata['created_at'],
            updated_at=doc_metadata['updated_at'],
            file_type=file_metadata['extension'],
            file_size=file_metadata['size'],
            language=language,
            description=description or summary,
            tags=tags or keywords,
            stats=DocumentStats(
                document_id=doc_id,
                word_count=stats['word_count'],
                character_count=stats['character_count'],
                chunk_count=len(chunks),
                page_count=metadata.get('page_count', 0) if metadata else 0,
                created_at=doc_metadata['created_at'],
                updated_at=doc_metadata['updated_at'],
                processing_time=time.time() - start_time,
                file_size=file_metadata['size'],
                embedding_model=settings.EMBEDDING_MODEL,
                embedding_dimension=settings.EMBEDDING_DIMENSION,
            )
        )
        
        logger.info(f"Document processed: {doc_id} - {doc_title}")
        
        return response
    
    async def process_text(self,
                      text: str,
                      title: str,
                      description: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> DocumentResponse:
        """Process a text document.
        
        Args:
            text: Document text
            title: Document title
            description: Document description (optional)
            tags: Document tags (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            DocumentResponse: Document response
        """
        start_time = time.time()
        
        # Generate document ID
        doc_id = generate_document_id(text, {'title': title})
        
        # Detect language
        language = detect_document_language(text)
        
        # Calculate document statistics
        stats = calculate_document_stats(text)
        
        # Split text into chunks
        chunks = await self._split_document_into_chunks(doc_id, text)
        
        # Generate embeddings for chunks
        await self._generate_embeddings_for_chunks(chunks)
        
        # Store chunks in vector store
        await self._store_chunks_in_vector_store(chunks)
        
        # Generate document summary
        summary = generate_document_summary(text)
        
        # Extract keywords
        keywords = extract_document_keywords(text)
        
        # Create document metadata
        doc_metadata = {
            'document_id': doc_id,
            'title': title,
            'source': 'text',
            'source_type': 'text',
            'file_path': None,
            'file_type': 'txt',
            'file_size': len(text.encode('utf-8')),
            'mime_type': 'text/plain',
            'language': language,
            'description': description or summary,
            'tags': tags or keywords,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'stats': {
                **stats,
                'chunk_count': len(chunks),
                'processing_time': time.time() - start_time,
                'embedding_model': settings.EMBEDDING_MODEL,
                'embedding_dimension': settings.EMBEDDING_DIMENSION,
            },
            'metadata': metadata or {},
        }
        
        # Save document metadata
        save_document_metadata(doc_id, doc_metadata)
        
        # Create document response
        response = DocumentResponse(
            document_id=doc_id,
            title=title,
            source='text',
            source_type='text',
            author=metadata.get('author') if metadata else None,
            created_at=doc_metadata['created_at'],
            updated_at=doc_metadata['updated_at'],
            file_type='txt',
            file_size=len(text.encode('utf-8')),
            language=language,
            description=description or summary,
            tags=tags or keywords,
            stats=DocumentStats(
                document_id=doc_id,
                word_count=stats['word_count'],
                character_count=stats['character_count'],
                chunk_count=len(chunks),
                page_count=1,
                created_at=doc_metadata['created_at'],
                updated_at=doc_metadata['updated_at'],
                processing_time=time.time() - start_time,
                file_size=len(text.encode('utf-8')),
                embedding_model=settings.EMBEDDING_MODEL,
                embedding_dimension=settings.EMBEDDING_DIMENSION,
            )
        )
        
        logger.info(f"Text document processed: {doc_id} - {title}")
        
        return response
    
    async def process_url(self,
                     url: str,
                     title: Optional[str] = None,
                     description: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> DocumentResponse:
        """Process a URL document.
        
        Args:
            url: Document URL
            title: Document title (optional)
            description: Document description (optional)
            tags: Document tags (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            DocumentResponse: Document response
        """
        # This is a placeholder for URL processing
        # In a real implementation, you would fetch the content from the URL
        # and process it similar to process_text or process_document
        
        # For now, we'll just return a dummy response
        doc_id = f"doc-{uuid.uuid4().hex[:16]}"
        
        response = DocumentResponse(
            document_id=doc_id,
            title=title or url,
            source=url,
            source_type='url',
            author=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            file_type='html',
            file_size=0,
            language='en',
            description=description or '',
            tags=tags or [],
            stats=DocumentStats(
                document_id=doc_id,
                word_count=0,
                character_count=0,
                chunk_count=0,
                page_count=0,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                processing_time=0,
                file_size=0,
                embedding_model=settings.EMBEDDING_MODEL,
                embedding_dimension=settings.EMBEDDING_DIMENSION,
            )
        )
        
        return response
    
    async def get_document(self, document_id: str) -> DocumentResponse:
        """Get a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            DocumentResponse: Document response
            
        Raises:
            DocumentNotFoundError: If the document does not exist
        """
        # Load document metadata
        metadata = load_document_metadata(document_id)
        
        # Create document response
        response = DocumentResponse(
            document_id=document_id,
            title=metadata.get('title', ''),
            source=metadata.get('source', ''),
            source_type=metadata.get('source_type', ''),
            author=metadata.get('metadata', {}).get('author'),
            created_at=metadata.get('created_at'),
            updated_at=metadata.get('updated_at'),
            file_type=metadata.get('file_type', ''),
            file_size=metadata.get('file_size', 0),
            language=metadata.get('language', 'en'),
            description=metadata.get('description', ''),
            tags=metadata.get('tags', []),
            stats=DocumentStats(
                document_id=document_id,
                word_count=metadata.get('stats', {}).get('word_count', 0),
                character_count=metadata.get('stats', {}).get('character_count', 0),
                chunk_count=metadata.get('stats', {}).get('chunk_count', 0),
                page_count=metadata.get('stats', {}).get('page_count', 0),
                created_at=metadata.get('created_at'),
                updated_at=metadata.get('updated_at'),
                processing_time=metadata.get('stats', {}).get('processing_time', 0),
                file_size=metadata.get('file_size', 0),
                embedding_model=metadata.get('stats', {}).get('embedding_model', settings.EMBEDDING_MODEL),
                embedding_dimension=metadata.get('stats', {}).get('embedding_dimension', settings.EMBEDDING_DIMENSION),
            )
        )
        
        return response
    
    async def list_documents(self) -> List[DocumentResponse]:
        """List all documents.
        
        Returns:
            List[DocumentResponse]: List of document responses
        """
        # List all documents
        documents = list_documents()
        
        # Create document responses
        responses = []
        for doc in documents:
            response = DocumentResponse(
                document_id=doc.get('document_id', ''),
                title=doc.get('title', ''),
                source=doc.get('source', ''),
                source_type=doc.get('source_type', ''),
                author=doc.get('metadata', {}).get('author'),
                created_at=doc.get('created_at'),
                updated_at=doc.get('updated_at'),
                file_type=doc.get('file_type', ''),
                file_size=doc.get('file_size', 0),
                language=doc.get('language', 'en'),
                description=doc.get('description', ''),
                tags=doc.get('tags', []),
                stats=DocumentStats(
                    document_id=doc.get('document_id', ''),
                    word_count=doc.get('stats', {}).get('word_count', 0),
                    character_count=doc.get('stats', {}).get('character_count', 0),
                    chunk_count=doc.get('stats', {}).get('chunk_count', 0),
                    page_count=doc.get('stats', {}).get('page_count', 0),
                    created_at=doc.get('created_at'),
                    updated_at=doc.get('updated_at'),
                    processing_time=doc.get('stats', {}).get('processing_time', 0),
                    file_size=doc.get('file_size', 0),
                    embedding_model=doc.get('stats', {}).get('embedding_model', settings.EMBEDDING_MODEL),
                    embedding_dimension=doc.get('stats', {}).get('embedding_dimension', settings.EMBEDDING_DIMENSION),
                )
            )
            responses.append(response)
        
        return responses
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            bool: True if document was deleted, False otherwise
            
        Raises:
            DocumentNotFoundError: If the document does not exist
        """
        # Delete document
        result = delete_document(document_id)
        
        # Delete document chunks from vector store
        if result and self.vector_store:
            try:
                await self.vector_store.delete_by_document_id(document_id)
            except Exception as e:
                logger.error(f"Error deleting document chunks from vector store: {document_id} - {e}")
        
        return result
    
    async def update_document(self, 
                        document_id: str, 
                        updates: Dict[str, Any]) -> DocumentResponse:
        """Update a document by ID.
        
        Args:
            document_id: Document ID
            updates: Document updates
            
        Returns:
            DocumentResponse: Updated document response
            
        Raises:
            DocumentNotFoundError: If the document does not exist
        """
        # Update document metadata
        metadata = update_document_metadata(document_id, updates)
        
        # Create document response
        response = DocumentResponse(
            document_id=document_id,
            title=metadata.title,
            source=metadata.source,
            source_type=metadata.source_type,
            author=metadata.metadata.get('author'),
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            file_type=metadata.file_type,
            file_size=metadata.file_size,
            language=metadata.language,
            description=metadata.description,
            tags=metadata.tags,
            stats=get_document_stats(document_id)
        )
        
        return response
    
    async def get_document_summary(self, document_id: str) -> DocumentSummary:
        """Get a document summary by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            DocumentSummary: Document summary
            
        Raises:
            DocumentNotFoundError: If the document does not exist
        """
        # Load document metadata
        metadata = load_document_metadata(document_id)
        
        # Create document summary
        summary = DocumentSummary(
            id=document_id,
            title=metadata.get('title', ''),
            summary=metadata.get('description', ''),
            key_points=[],
            entities={},
            topics=metadata.get('tags', [])
        )
        
        return summary
    
    async def search_documents(self, 
                         query: str, 
                         document_id: Optional[str] = None,
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents by query.
        
        Args:
            query: Search query
            document_id: Document ID (optional)
            top_k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results
            
        Raises:
            VectorStoreError: If there is an error searching the vector store
        """
        if not self.vector_store:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Search vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                document_id=document_id,
                top_k=top_k
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise VectorStoreError(f"Error searching documents: {e}")
    
    async def _extract_text_from_document(self, file_path: str, file_type: str) -> str:
        """Extract text from a document.
        
        Args:
            file_path: Path to the document file
            file_type: Document file type
            
        Returns:
            str: Extracted text
        """
        # This is a placeholder for text extraction
        # In a real implementation, you would use different methods based on file_type
        # For example, PyPDF2 for PDF, python-docx for DOCX, etc.
        
        # For now, we'll just return a dummy text
        return "This is a placeholder for extracted text from a document."
    
    async def _split_document_into_chunks(self, doc_id: str, text: str) -> List[DocumentChunk]:
        """Split a document into chunks.
        
        Args:
            doc_id: Document ID
            text: Document text
            
        Returns:
            List[DocumentChunk]: List of document chunks
        """
        # Split text into chunks
        text_chunks = split_text_into_chunks(
            text=text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Create document chunks
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{doc_id}-chunk-{i}"
            chunk = DocumentChunk(
                id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                metadata={
                    'chunk_index': i,
                    'document_id': doc_id,
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_embeddings_for_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            
        Raises:
            EmbeddingGenerationError: If there is an error generating embeddings
        """
        if not self.embedding_service:
            raise EmbeddingGenerationError("Embedding service not initialized")
        
        try:
            for chunk in chunks:
                # Generate embedding for chunk
                embedding = await self.embedding_service.generate_embedding(chunk.text)
                chunk.embedding = embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise EmbeddingGenerationError(f"Error generating embeddings: {e}")
    
    async def _store_chunks_in_vector_store(self, chunks: List[DocumentChunk]) -> None:
        """Store document chunks in the vector store.
        
        Args:
            chunks: List of document chunks
            
        Raises:
            VectorStoreError: If there is an error storing chunks in the vector store
        """
        if not self.vector_store:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            # Store chunks in vector store
            await self.vector_store.add_documents(chunks)
        except Exception as e:
            logger.error(f"Error storing chunks in vector store: {e}")
            raise VectorStoreError(f"Error storing chunks in vector store: {e}")