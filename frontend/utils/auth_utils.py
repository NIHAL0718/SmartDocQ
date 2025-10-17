"""Authentication utility functions for SmartDocQ."""

import requests
import streamlit as st
import json
import os
from typing import Dict, Any, Optional, Tuple

# API endpoint
API_URL = "http://localhost:8000/api"

# File to store authentication data
AUTH_FILE = "auth_data.json"

def save_auth_data(user_data: Dict[str, Any], token: str):
    """Save authentication data to file for persistence."""
    try:
        auth_data = {
            "user": user_data,
            "token": token,
            "authenticated": True
        }
        with open(AUTH_FILE, 'w') as f:
            json.dump(auth_data, f)
    except Exception as e:
        print(f"Error saving auth data: {e}")

def load_auth_data() -> Optional[Dict[str, Any]]:
    """Load authentication data from file."""
    try:
        if os.path.exists(AUTH_FILE):
            with open(AUTH_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading auth data: {e}")
    return None

def clear_auth_data():
    """Clear authentication data from file."""
    try:
        if os.path.exists(AUTH_FILE):
            os.remove(AUTH_FILE)
    except Exception as e:
        print(f"Error clearing auth data: {e}")

def verify_token(token: str) -> bool:
    """Verify if the stored token is still valid."""
    try:
        # Make a simple request to verify token
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{API_URL}/health", headers=headers, timeout=5)
        return response.status_code == 200
    except:
        return False


def login_user(username: str, password: str) -> Tuple[bool, str]:
    """Login user with username and password.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        Tuple of (success, message)
    """
    try:
        response = requests.post(
            f"{API_URL}/auth/login/json",
            json={"username": username, "password": password},
            timeout=6
        )
        
        if response.status_code == 200:
            data = response.json()
            # Store user data and token in session state
            if data.get("success", False):
                # Create a user object since the simple_server returns different format
                user_data = {
                    "username": username,
                    "user_id": data.get("user_id", ""),
                }
                token = data.get("token", "")
                
                # Store in session state
                st.session_state.user = user_data
                st.session_state.token = token
                st.session_state.authenticated = True
                
                # Save to file for persistence
                save_auth_data(user_data, token)
                
                return True, "Login successful"
            else:
                return False, data.get("message", "Login failed")
        else:
            error_msg = response.json().get("detail", "Login failed")
            return False, error_msg
    except Exception as e:
        return False, f"Error: {str(e)}"


def register_user(username: str, password: str, email: Optional[str] = None) -> Tuple[bool, str]:
    """Register a new user.
    
    Args:
        username: Username
        password: Password
        email: Optional email
        
    Returns:
        Tuple of (success, message)
    """
    try:
        user_data = {"username": username, "password": password}
        if email:
            user_data["email"] = email
            
        response = requests.post(
            f"{API_URL}/auth/register",
            json=user_data,
            timeout=8
        )
        
        if response.status_code in [200, 201]:
            data = response.json()
            if data.get("success", False):
                return True, data.get("message", "Registration successful")
            else:
                return False, data.get("message", "Registration failed")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", "Registration failed")
            except ValueError:
                # Handle JSON parsing error
                error_msg = f"Registration failed with status code: {response.status_code}"
            return False, error_msg
    except Exception as e:
        return False, f"Error: {str(e)}"


def logout_user() -> None:
    """Logout current user."""
    st.session_state.user = None
    st.session_state.token = None
    st.session_state.authenticated = False
    clear_auth_data()


def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current authenticated user.
    
    Returns:
        User data or None if not authenticated
    """
    if st.session_state.authenticated and st.session_state.user:
        return st.session_state.user
    return None


def is_authenticated() -> bool:
    """Check if user is authenticated.
    
    Returns:
        True if authenticated, False otherwise
    """
    # First check session state
    if hasattr(st.session_state, 'authenticated') and st.session_state.authenticated:
        return True
    
    # If not in session state, try to load from file
    auth_data = load_auth_data()
    if auth_data and auth_data.get("authenticated", False):
        token = auth_data.get("token")
        if token and verify_token(token):
            # Restore session state
            st.session_state.user = auth_data.get("user")
            st.session_state.token = token
            st.session_state.authenticated = True
            return True
        else:
            # Token is invalid, clear the file
            clear_auth_data()
    
    return False


def get_auth_header() -> Dict[str, str]:
    """Get authentication header for API requests.
    
    Returns:
        Authentication header dict
    """
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}