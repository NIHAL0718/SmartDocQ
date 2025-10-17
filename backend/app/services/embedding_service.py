"""Service for generating embeddings for document chunks."""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.models.document import DocumentChunk

# Load environment variables
load_dotenv()

# Initialize the embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def generate_embeddings(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Generate embeddings for a list of document chunks.
    
    Args:
        chunks: List of DocumentChunk objects
        
    Returns:
        List of DocumentChunk objects with embeddings added
    """
    if not chunks:
        return []
    
    # Extract text from chunks
    texts = [chunk.text for chunk in chunks]
    
    # Generate embeddings
    embeddings = embedding_model.encode(texts)
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i].tolist()
    
    return chunks


def generate_query_embedding(query: str) -> List[float]:
    """Generate embedding for a query string.
    
    Args:
        query: Query text
        
    Returns:
        Embedding vector as a list of floats
    """
    # Generate embedding
    embedding = embedding_model.encode(query)
    
    return embedding.tolist()


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Convert to numpy arrays
    vec1 = np.array(embedding1).reshape(1, -1)
    vec2 = np.array(embedding2).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vec1, vec2)[0][0]
    
    return float(similarity)


def batch_generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    # Generate embeddings
    embeddings = embedding_model.encode(texts)
    
    # Convert to list of lists
    return [embedding.tolist() for embedding in embeddings]


def get_embedding_model_info() -> Dict[str, Any]:
    """Get information about the current embedding model.
    
    Returns:
        Dictionary with model information
    """
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "embedding_dimension": embedding_model.get_sentence_embedding_dimension(),
        "max_sequence_length": embedding_model.max_seq_length,
    }