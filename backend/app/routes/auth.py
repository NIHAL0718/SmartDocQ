"""Authentication routes for user login and registration."""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Dict, Any

from app.models.user import UserCreate, UserResponse, Token
from app.services.user_service import create_user, authenticate_user
from app.core.logging import get_logger

# Initialize router
router = APIRouter()

# Initialize logger
logger = get_logger("auth_router")

# OAuth2 password bearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user.
    
    Args:
        user_data: User data for registration
        
    Returns:
        Created user
        
    Raises:
        HTTPException: If username already exists
    """
    user = create_user(user_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists"
        )
    return user


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with username and password.
    
    Args:
        form_data: Form data with username and password
        
    Returns:
        Access token
        
    Raises:
        HTTPException: If authentication fails
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "access_token": user["access_token"],
        "token_type": user["token_type"]
    }


@router.post("/login/json")
async def login_json(data: Dict[str, str]):
    """Login with username and password in JSON format.
    
    Args:
        data: JSON data with username and password
        
    Returns:
        User data with access token
        
    Raises:
        HTTPException: If authentication fails
    """
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required"
        )
    
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user