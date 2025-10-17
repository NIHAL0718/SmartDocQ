"""Database utility functions for MongoDB operations."""

import logging
from typing import Dict, Any, List, Optional
from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

from app.core.config import settings
from app.core.logging import get_logger

# Initialize logger
logger = get_logger("db_utils")

# MongoDB client
client = MongoClient(settings.MONGODB_URI)
db = client[settings.MONGODB_DB_NAME]


def get_collection(collection_name: str) -> Collection:
    """Get MongoDB collection.
    
    Args:
        collection_name: Collection name
        
    Returns:
        MongoDB collection
    """
    return db[collection_name]


def insert_one(collection_name: str, document: Dict[str, Any]) -> str:
    """Insert a document into a collection.
    
    Args:
        collection_name: Collection name
        document: Document to insert
        
    Returns:
        Inserted document ID
    """
    collection = get_collection(collection_name)
    result = collection.insert_one(document)
    logger.debug(f"Inserted document with ID {result.inserted_id} into {collection_name}")
    return str(result.inserted_id)


def find_one(collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a document in a collection.
    
    Args:
        collection_name: Collection name
        query: Query filter
        
    Returns:
        Found document or None
    """
    collection = get_collection(collection_name)
    result = collection.find_one(query)
    return result


def update_one(collection_name: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
    """Update a document in a collection.
    
    Args:
        collection_name: Collection name
        query: Query filter
        update: Update operations
        
    Returns:
        True if document was updated, False otherwise
    """
    collection = get_collection(collection_name)
    result = collection.update_one(query, update)
    success = result.modified_count > 0
    if success:
        logger.debug(f"Updated document in {collection_name}")
    return success


def delete_one(collection_name: str, query: Dict[str, Any]) -> bool:
    """Delete a document from a collection.
    
    Args:
        collection_name: Collection name
        query: Query filter
        
    Returns:
        True if document was deleted, False otherwise
    """
    collection = get_collection(collection_name)
    result = collection.delete_one(query)
    success = result.deleted_count > 0
    if success:
        logger.debug(f"Deleted document from {collection_name}")
    return success


def find_many(collection_name: str, query: Dict[str, Any], limit: int = 0, skip: int = 0, sort: List[tuple] = None) -> List[Dict[str, Any]]:
    """Find multiple documents in a collection.
    
    Args:
        collection_name: Collection name
        query: Query filter
        limit: Maximum number of documents to return (0 for no limit)
        skip: Number of documents to skip
        sort: List of (key, direction) tuples for sorting
        
    Returns:
        List of found documents
    """
    collection = get_collection(collection_name)
    cursor = collection.find(query)
    
    if skip > 0:
        cursor = cursor.skip(skip)
    
    if limit > 0:
        cursor = cursor.limit(limit)
    
    if sort:
        cursor = cursor.sort(sort)
    
    return list(cursor)