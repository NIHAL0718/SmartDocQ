"""User models for authentication and user management."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr


class UserBase(BaseModel):
    """Base user model with common attributes."""
    username: str
    email: Optional[str] = None


class UserCreate(UserBase):
    """User creation model with password."""
    password: str


class UserInDB(UserBase):
    """User model as stored in the database."""
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_admin: bool = False


class UserResponse(UserBase):
    """User model for API responses."""
    id: str
    created_at: datetime
    is_admin: bool = False

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token model for authentication."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model for JWT payload."""
    sub: str
    username: str
    is_admin: bool = False