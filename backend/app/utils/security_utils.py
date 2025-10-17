"""Utility functions for security operations."""

import os
import uuid
import hashlib
import secrets
import string
from typing import Optional, Tuple
from datetime import datetime, timedelta

import jwt
from passlib.context import CryptContext

from ..core.logging import get_logger
from ..core.config import settings

# Initialize logger
logger = get_logger("security_utils")

# Initialize password context using PBKDF2 to avoid bcrypt length limits and binary wheels issues
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def generate_password_hash(password: str) -> str:
    """Generate a password hash.
    
    Args:
        password (str): Plain text password
        
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.
    
    Args:
        plain_password (str): Plain text password
        hashed_password (str): Hashed password
        
    Returns:
        bool: True if password matches hash, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def generate_token(data: dict, expires_delta: Optional[timedelta] = None, secret_key: Optional[str] = None) -> str:
    """
Generate a JWT token.
    
    Args:
        data (dict): Data to encode in the token
        expires_delta (Optional[timedelta]): Token expiration time
        secret_key (Optional[str]): Secret key for token signing
        
    Returns:
        str: JWT token
    """
    # Create a copy of the data
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    # Use provided secret key or default
    if secret_key is None:
        secret_key = settings.SECRET_KEY
    
    # Encode token
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    
    return encoded_jwt


def decode_token(token: str, secret_key: Optional[str] = None) -> dict:
    """Decode a JWT token.
    
    Args:
        token (str): JWT token
        secret_key (Optional[str]): Secret key for token verification
        
    Returns:
        dict: Decoded token data
        
    Raises:
        jwt.PyJWTError: If token is invalid
    """
    # Use provided secret key or default
    if secret_key is None:
        secret_key = settings.SECRET_KEY
    
    # Decode token
    decoded_jwt = jwt.decode(token, secret_key, algorithms=["HS256"])
    
    return decoded_jwt


def generate_api_key() -> str:
    """Generate a random API key.
    
    Returns:
        str: API key
    """
    # Generate a random UUID
    api_key = f"sk-{uuid.uuid4().hex}"
    
    return api_key


def generate_random_password(length: int = 12) -> str:
    """Generate a random password.
    
    Args:
        length (int): Password length
        
    Returns:
        str: Random password
    """
    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = string.punctuation
    
    # Ensure at least one character from each set
    password = [
        secrets.choice(lowercase),
        secrets.choice(uppercase),
        secrets.choice(digits),
        secrets.choice(special)
    ]
    
    # Fill the rest of the password
    all_chars = lowercase + uppercase + digits + special
    password.extend(secrets.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle the password
    secrets.SystemRandom().shuffle(password)
    
    # Convert to string
    password_str = ''.join(password)
    
    return password_str


def generate_random_string(length: int = 16, include_digits: bool = True, include_special: bool = False) -> str:
    """Generate a random string.
    
    Args:
        length (int): String length
        include_digits (bool): Whether to include digits
        include_special (bool): Whether to include special characters
        
    Returns:
        str: Random string
    """
    # Define character sets
    chars = string.ascii_letters
    
    if include_digits:
        chars += string.digits
    
    if include_special:
        chars += string.punctuation
    
    # Generate random string
    random_string = ''.join(secrets.choice(chars) for _ in range(length))
    
    return random_string


def hash_data(data: str, algorithm: str = 'sha256') -> str:
    """Hash data using the specified algorithm.
    
    Args:
        data (str): Data to hash
        algorithm (str): Hash algorithm
        
    Returns:
        str: Hashed data
    """
    # Convert data to bytes
    data_bytes = data.encode('utf-8')
    
    # Hash data
    if algorithm == 'md5':
        hash_obj = hashlib.md5(data_bytes)
    elif algorithm == 'sha1':
        hash_obj = hashlib.sha1(data_bytes)
    elif algorithm == 'sha256':
        hash_obj = hashlib.sha256(data_bytes)
    elif algorithm == 'sha512':
        hash_obj = hashlib.sha512(data_bytes)
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Return hexadecimal digest
    return hash_obj.hexdigest()


def generate_salt(length: int = 16) -> str:
    """Generate a random salt.
    
    Args:
        length (int): Salt length
        
    Returns:
        str: Random salt
    """
    return secrets.token_hex(length)


def hash_password_with_salt(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash a password with a salt.
    
    Args:
        password (str): Password to hash
        salt (Optional[str]): Salt to use (if None, a new salt will be generated)
        
    Returns:
        Tuple[str, str]: Tuple of (hashed_password, salt)
    """
    # Generate salt if not provided
    if salt is None:
        salt = generate_salt()
    
    # Convert salt to bytes
    salt_bytes = salt.encode('utf-8')
    
    # Hash password with salt
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt_bytes, 100000)
    hashed_password = hash_obj.hex()
    
    return hashed_password, salt


def verify_password_with_salt(password: str, hashed_password: str, salt: str) -> bool:
    """Verify a password against a hash and salt.
    
    Args:
        password (str): Password to verify
        hashed_password (str): Hashed password
        salt (str): Salt used for hashing
        
    Returns:
        bool: True if password matches hash, False otherwise
    """
    # Hash the password with the salt
    calculated_hash, _ = hash_password_with_salt(password, salt)
    
    # Compare the hashes
    return calculated_hash == hashed_password


def generate_secure_filename(filename: str) -> str:
    """Generate a secure filename.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Secure filename
    """
    # Get file extension
    _, ext = os.path.splitext(filename)
    
    # Generate a random filename
    secure_name = f"{uuid.uuid4().hex}{ext}"
    
    return secure_name


def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent XSS attacks.
    
    Args:
        input_str (str): Input string
        
    Returns:
        str: Sanitized string
    """
    # Replace potentially dangerous characters
    sanitized = input_str.replace('<', '&lt;').replace('>', '&gt;')
    
    return sanitized


def generate_csrf_token() -> str:
    """Generate a CSRF token.
    
    Returns:
        str: CSRF token
    """
    return secrets.token_hex(32)


def generate_session_id() -> str:
    """Generate a session ID.
    
    Returns:
        str: Session ID
    """
    return secrets.token_urlsafe(32)


def is_valid_api_key(api_key: str) -> bool:
    """Check if an API key is valid.
    
    Args:
        api_key (str): API key to check
        
    Returns:
        bool: True if API key is valid, False otherwise
    """
    # Check if API key starts with 'sk-'
    if not api_key.startswith('sk-'):
        return False
    
    # Check if API key has the correct length
    if len(api_key) != 35:  # 'sk-' + 32 hex characters
        return False
    
    # Check if API key contains only valid characters
    valid_chars = set(string.hexdigits + '-')
    if not all(c in valid_chars for c in api_key):
        return False
    
    # In a real application, you would check if the API key exists in the database
    # For now, we'll just check if it matches the configured API key
    return api_key == settings.API_KEY