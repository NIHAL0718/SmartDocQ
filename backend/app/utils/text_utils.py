"""Utility functions for text processing."""

import re
import string
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from ..core.logging import get_logger

# Initialize logger
logger = get_logger("text_utils")


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace, converting to lowercase, etc.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    return text


def clean_text(text: str, remove_punctuation: bool = True, remove_numbers: bool = False) -> str:
    """Clean text by removing punctuation, numbers, etc.
    
    Args:
        text (str): Text to clean
        remove_punctuation (bool): Whether to remove punctuation
        remove_numbers (bool): Whether to remove numbers
        
    Returns:
        str: Cleaned text
    """
    # Normalize text first
    text = normalize_text(text)
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text


def extract_keywords(text: str, max_keywords: int = 10, min_word_length: int = 3) -> List[str]:
    """Extract keywords from text.
    
    Args:
        text (str): Text to extract keywords from
        max_keywords (int): Maximum number of keywords to extract
        min_word_length (int): Minimum word length
        
    Returns:
        List[str]: List of keywords
    """
    # Clean the text
    cleaned_text = clean_text(text, remove_punctuation=True, remove_numbers=True)
    
    # Split into words
    words = cleaned_text.split()
    
    # Filter out short words and common stop words
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'when', 'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through',
        'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on',
        'at', 'by', 'with', 'about', 'against', 'between', 'into', 'through',
        'after', 'before', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'too', 'very', 'can', 'will', 'should', 'now'
    }
    
    filtered_words = [word for word in words if len(word) >= min_word_length and word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get the most common words
    keywords = [word for word, _ in word_counts.most_common(max_keywords)]
    
    return keywords


def calculate_text_stats(text: str) -> Dict[str, Any]:
    """Calculate statistics for text.
    
    Args:
        text (str): Text to calculate statistics for
        
    Returns:
        Dict[str, Any]: Text statistics
    """
    # Normalize text
    normalized_text = normalize_text(text)
    
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', normalized_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Split into words
    words = re.findall(r'\b\w+\b', normalized_text)
    
    # Calculate statistics
    stats = {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'average_sentence_length': sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0,
    }
    
    return stats


def split_text_by_sentences(text: str) -> List[str]:
    """Split text into sentences.
    
    Args:
        text (str): Text to split
        
    Returns:
        List[str]: List of sentences
    """
    # Simple sentence splitting (can be improved with more sophisticated NLP)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def split_text_by_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.
    
    Args:
        text (str): Text to split
        
    Returns:
        List[str]: List of paragraphs
    """
    # Split by double newlines (common paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks of a specified size.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Normalize text
    text = normalize_text(text)
    
    # Check if text is shorter than chunk size
    if len(text) <= chunk_size:
        return [text]
    
    # Split text into chunks
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk of text
        end = start + chunk_size
        chunk = text[start:end]
        
        # If this is not the last chunk, try to end at a sentence boundary
        if end < len(text):
            # Find the last sentence boundary in the chunk
            last_boundary = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
            
            # If a sentence boundary was found, end the chunk there
            if last_boundary != -1 and last_boundary > chunk_size // 2:
                end = start + last_boundary + 1
                chunk = text[start:end]
        
        # Add chunk to list
        chunks.append(chunk.strip())
        
        # Move start position for next chunk (with overlap)
        start = end - chunk_overlap
    
    return chunks


def detect_language(text: str) -> str:
    """Detect the language of text (simple implementation).
    
    Args:
        text (str): Text to detect language for
        
    Returns:
        str: Detected language code (ISO 639-1)
    """
    # This is a very simple implementation that only checks for a few languages
    # For production use, consider using a proper language detection library
    
    # Normalize and clean text
    text = normalize_text(text)
    
    # Sample of common words in different languages
    language_words = {
        'en': {'the', 'and', 'is', 'in', 'to', 'it', 'of', 'that', 'you', 'for'},
        'es': {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'ser', 'se', 'no'},
        'fr': {'le', 'la', 'de', 'et', 'est', 'en', 'un', 'que', 'qui', 'pour'},
        'de': {'der', 'die', 'das', 'und', 'ist', 'in', 'zu', 'den', 'mit', 'nicht'},
    }
    
    # Split text into words
    words = set(re.findall(r'\b\w+\b', text.lower()))
    
    # Count matches for each language
    matches = {}
    for lang, lang_words in language_words.items():
        matches[lang] = len(words.intersection(lang_words))
    
    # Return the language with the most matches
    if not matches or max(matches.values()) == 0:
        return 'unknown'
    
    return max(matches, key=matches.get)


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Generate a simple extractive summary of text.
    
    Args:
        text (str): Text to summarize
        max_sentences (int): Maximum number of sentences in the summary
        
    Returns:
        str: Summary text
    """
    # Split text into sentences
    sentences = split_text_by_sentences(text)
    
    # If there are fewer sentences than the maximum, return the original text
    if len(sentences) <= max_sentences:
        return text
    
    # Simple approach: take the first sentence and the last few sentences
    summary_sentences = [sentences[0]]
    
    # Add sentences from the middle and end
    if max_sentences > 1:
        middle_idx = len(sentences) // 2
        summary_sentences.append(sentences[middle_idx])
    
    if max_sentences > 2:
        remaining = max_sentences - 2
        step = len(sentences) // (remaining + 1)
        for i in range(1, remaining + 1):
            idx = i * step
            if idx < len(sentences) and sentences[idx] not in summary_sentences:
                summary_sentences.append(sentences[idx])
    
    # Join sentences into a summary
    summary = ' '.join(summary_sentences)
    
    return summary


def find_similar_texts(query: str, texts: List[str], top_n: int = 3) -> List[Tuple[int, float]]:
    """Find texts similar to a query using simple TF-IDF and cosine similarity.
    
    Args:
        query (str): Query text
        texts (List[str]): List of texts to search
        top_n (int): Number of top results to return
        
    Returns:
        List[Tuple[int, float]]: List of (text_index, similarity_score) tuples
    """
    # Clean and normalize query and texts
    query = clean_text(query)
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Split into words
    query_words = set(query.split())
    text_words = [set(text.split()) for text in cleaned_texts]
    
    # Calculate similarity scores (simple Jaccard similarity)
    scores = []
    for i, words in enumerate(text_words):
        if not words or not query_words:
            scores.append((i, 0.0))
            continue
        
        # Jaccard similarity: intersection / union
        intersection = len(query_words.intersection(words))
        union = len(query_words.union(words))
        similarity = intersection / union if union > 0 else 0.0
        
        scores.append((i, similarity))
    
    # Sort by similarity score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N results
    return scores[:top_n]