"""Authentication utility functions for SmartDocQ."""

import os
import requests
import streamlit as st
from typing import Dict, Any, Optional, Tuple

# API endpoint
API_URL = "https://smartdocq.onrender.com/api"


def login_user(username: str, password: str) -> Tuple[bool, str]:
    """Login user with username and password.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Log login attempt for debugging
        print(f"Attempting to login user: {username} with MongoDB Atlas")
        
        # Add timeout to prevent hanging
        response = requests.post(
            f"{API_URL}/auth/login/json",
            json={"username": username, "password": password},
            timeout=30
        )
        
        # Print response for debugging
        print(f"Login response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # Store user data and token in session state
            if data.get("success", False):
                # Create a user object since the simple_server returns different format
                st.session_state.user = {
                    "username": username,
                    "user_id": data.get("user_id", ""),
                }
                st.session_state.token = data.get("token", "")
                st.session_state.authenticated = True
                
                # Save authentication data to file for persistence
                import json
                import os
                auth_data = {
                    "user": st.session_state.user,
                    "token": st.session_state.token,
                    "authenticated": True
                }
                with open("auth_data.json", "w") as f:
                    json.dump(auth_data, f)
                
                return True, "Login successful! Welcome to SmartDocQ."
            else:
                print(f"Login failed: {data.get('message', 'Unknown error')}")
                return False, data.get("message", "Login failed. Please check your credentials.")
        else:
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_msg = error_data.get("detail")
                elif "message" in error_data:
                    error_msg = error_data.get("message")
                else:
                    error_msg = "Login failed. Please check your credentials."
            except ValueError:
                error_msg = f"Login failed with status code: {response.status_code}"
            
            print(f"Login error: {error_msg}")
            return False, error_msg
    except requests.exceptions.ConnectionError:
        return False, "Connection error. Please check if the backend server is running."
    except requests.exceptions.Timeout:
        return False, "Connection timed out. The server might be busy, please try again."
    except Exception as e:
        print(f"Unexpected login error: {str(e)}")
        return False, f"Login error: {str(e)}"


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
        
        # Log registration attempt for debugging
        print(f"Attempting to register user: {username} with MongoDB Atlas")
        
        # Add timeout to prevent hanging
        response = requests.post(
            f"{API_URL}/auth/register",
            json=user_data,
            timeout=30
        )
        
        # Print response for debugging
        print(f"Registration response status: {response.status_code}")
        
        if response.status_code == 201 or (response.status_code == 200 and response.json().get("success", False)):
            return True, "Registration successful! You can now log in with your credentials."
        else:
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_msg = error_data.get("detail")
                elif "message" in error_data:
                    error_msg = error_data.get("message")
                else:
                    error_msg = "Registration failed. Please try again."
            except ValueError:
                # Handle JSON parsing error
                error_msg = f"Registration failed with status code: {response.status_code}. Please try again."
            
            print(f"Registration error: {error_msg}")
            return False, error_msg
    except requests.exceptions.ConnectionError:
        return False, "Connection error. Please check if the backend server is running."
    except requests.exceptions.Timeout:
        return False, "Connection timed out. The server might be busy, please try again."
    except Exception as e:
        print(f"Unexpected registration error: {str(e)}")
        return False, f"Registration error: {str(e)}"


def logout_user() -> None:
    """Logout current user."""
    st.session_state.user = None
    st.session_state.token = None
    st.session_state.authenticated = False


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
    return st.session_state.authenticated


def get_auth_header() -> Dict[str, str]:
    """Get authentication header for API requests.
    
    Returns:
        Authentication header dict
    """
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}