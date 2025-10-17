"""Service for vector database operations."""

import os
import time
from typing import List, Dict, Any, Optional
import chromadb

from app.models.document import DocumentChunk
from app.models.qa import SourceChunk
from app.services.embedding_service import generate_query_embedding

# Initialize ChromaDB client
PERSISTENCE_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma")
os.makedirs(PERSISTENCE_DIRECTORY, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=PERSISTENCE_DIRECTORY)

# Create or get the collection
chroma_collection = chroma_client.get_or_create_collection("document_chunks")


def store_document_chunks(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Store document chunks in the vector database.
    
    Args:
        chunks: List of DocumentChunk objects with embeddings
        
    Returns:
        Dictionary with storage results
    """
    if not chunks:
        return {"status": "error", "message": "No chunks provided"}
    
    try:
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadatas = [{
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.metadata.get("chunk_index", 0),
            "source": chunk.metadata.get("source", ""),
        } for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        
        # Add to collection
        chroma_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        # Persist changes
        chroma_client.persist()
        
        return {
            "status": "success",
            "message": f"Stored {len(chunks)} chunks in vector database",
            "chunk_count": len(chunks),
        }
    
    except Exception as e:
        # Log the error
        print(f"Error storing chunks in vector database: {str(e)}")
        
        return {
            "status": "error",
            "message": f"Failed to store chunks: {str(e)}",
        }


def search_similar_chunks(query: str, doc_id: Optional[str] = None, limit: int = 5) -> List[SourceChunk]:
    """Search for document chunks similar to the query.
    
    Args:
        query: Query text
        doc_id: Optional document ID to filter results
        limit: Maximum number of results to return
        
    Returns:
        List of SourceChunk objects
    """
    try:
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        # Prepare filter if doc_id is provided
        where_filter = {"doc_id": doc_id} if doc_id else None
        
        # Query the collection
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        source_chunks = []
        for i in range(len(results["ids"][0])):
            # Convert distance to similarity score (1 - distance)
            similarity = 1.0 - min(1.0, max(0.0, results["distances"][0][i]))
            
            source_chunks.append(
                SourceChunk(
                    text=results["documents"][0][i],
                    source=results["metadatas"][0][i].get("source", ""),
                    relevance_score=similarity,
                    doc_id=results["metadatas"][0][i].get("doc_id"),
                    chunk_id=results["ids"][0][i],
                )
            )
        
        return source_chunks
    
    except Exception as e:
        # Log the error
        print(f"Error searching vector database: {str(e)}")
        
        # Return empty list on error
        return []


def get_document_chunks(doc_id: str) -> List[DocumentChunk]:
    """Get all chunks for a specific document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        List of DocumentChunk objects
    """
    try:
        # Query the collection for all chunks of the document
        results = chroma_collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "embeddings"]
        )
        
        # Process results
        chunks = []
        for i in range(len(results["ids"])):
            chunk_id = results["ids"][i]
            metadata = results["metadatas"][i]
            text = results["documents"][i]
            embedding = results["embeddings"][i]
            
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    metadata=metadata,
                    embedding=embedding,
                )
            )
        
        return chunks
    
    except Exception as e:
        # Log the error
        print(f"Error getting document chunks: {str(e)}")
        
        # Return empty list on error
        return []


def delete_document_chunks(doc_id: str) -> Dict[str, Any]:
    """Delete all chunks for a specific document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Dictionary with deletion results
    """
    try:
        # Delete chunks with the specified doc_id
        chroma_collection.delete(where={"doc_id": doc_id})
        
        # Persist changes
        chroma_client.persist()
        
        return {
            "status": "success",
            "message": f"Deleted all chunks for document {doc_id}",
        }
    
    except Exception as e:
        # Log the error
        print(f"Error deleting document chunks: {str(e)}")
        
        return {
            "status": "error",
            "message": f"Failed to delete chunks: {str(e)}",
        }


def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about the vector database collection.
    
    Returns:
        Dictionary with collection statistics
    """
    try:
        # Get collection count
        count = chroma_collection.count()
        
        # Get unique document IDs
        results = chroma_collection.get(
            include=["metadatas"],
            limit=10000  # Set a reasonable limit
        )
        
        doc_ids = set()
        for metadata in results["metadatas"]:
            doc_ids.add(metadata.get("doc_id"))
        
        return {
            "status": "success",
            "total_chunks": count,
            "total_documents": len(doc_ids),
            "document_ids": list(doc_ids),
        }
    
    except Exception as e:
        # Log the error
        print(f"Error getting collection stats: {str(e)}")
        
        return {
            "status": "error",
            "message": f"Failed to get collection stats: {str(e)}",
        }