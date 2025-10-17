"""User service for handling user-related operations."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from bson import ObjectId

from app.core.config import settings
from app.models.user import UserCreate, UserInDB, UserResponse
from app.utils.db_utils import insert_one, find_one, update_one
from app.utils.security_utils import generate_password_hash, verify_password, generate_token

logger = logging.getLogger(__name__)


def create_user(user_data: UserCreate) -> Optional[UserResponse]:
    """Create a new user.
    
    Args:
        user_data: User data for creation
        
    Returns:
        Created user or None if username already exists
    """
    # Check if username already exists
    existing_user = find_one(settings.MONGODB_USER_COLLECTION, {"username": user_data.username})
    if existing_user:
        logger.warning(f"User with username {user_data.username} already exists")
        return None
    
    # Check if email already exists (if provided)
    if user_data.email:
        existing_email = find_one(settings.MONGODB_USER_COLLECTION, {"email": user_data.email})
        if existing_email:
            logger.warning(f"User with email {user_data.email} already exists")
            return None
    
    # Create user in database
    hashed_password = generate_password_hash(user_data.password)
    user_in_db = UserInDB(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    
    # Convert to dict and insert
    user_dict = user_in_db.model_dump()
    user_id = insert_one(settings.MONGODB_USER_COLLECTION, user_dict)
    
    # Return user response
    return UserResponse(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        created_at=user_in_db.created_at,
        is_admin=user_in_db.is_admin
    )


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        User data with token if authentication successful, None otherwise
    """
    user = find_one(settings.MONGODB_USER_COLLECTION, {"username": username})
    if not user:
        logger.warning(f"User with username {username} not found")
        return None
    
    if not verify_password(password, user["hashed_password"]):
        logger.warning(f"Invalid password for user {username}")
        return None
    
    # Generate token
    token_data = {
        "sub": str(user["_id"]),
        "username": user["username"],
        "is_admin": user.get("is_admin", False)
    }
    token = generate_token(token_data)
    
    # Create user response
    user_response = UserResponse(
        id=str(user["_id"]),
        username=user["username"],
        email=user.get("email"),
        created_at=user["created_at"],
        is_admin=user.get("is_admin", False)
    )
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_response.model_dump()
    }


def get_user_by_id(user_id: str) -> Optional[UserResponse]:
    """Get user by ID.
    
    Args:
        user_id: User ID
        
    Returns:
        User data if found, None otherwise
    """
    try:
        user = find_one(settings.MONGODB_USER_COLLECTION, {"_id": ObjectId(user_id)})
        if not user:
            logger.warning(f"User with ID {user_id} not found")
            return None
        
        return UserResponse(
            id=str(user["_id"]),
            username=user["username"],
            email=user.get("email"),
            created_at=user["created_at"],
            is_admin=user.get("is_admin", False)
        )
    except Exception as e:
        logger.error(f"Error getting user by ID: {e}")
        return None


def get_user_by_username(username: str) -> Optional[UserResponse]:
    """Get user by username.
    
    Args:
        username: Username
        
    Returns:
        User data if found, None otherwise
    """
    user = find_one(settings.MONGODB_USER_COLLECTION, {"username": username})
    if not user:
        logger.warning(f"User with username {username} not found")
        return None
    
    return UserResponse(
        id=str(user["_id"]),
        username=user["username"],
        email=user.get("email"),
        created_at=user["created_at"],
        is_admin=user.get("is_admin", False)
    )