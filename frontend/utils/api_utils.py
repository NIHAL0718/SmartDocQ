"""Utility functions for making API calls with retry logic and proper timeout handling."""

import requests
import time
import streamlit as st
from typing import Optional, Dict, Any, Tuple


def make_api_call_with_retry(
    method: str,
    url: str,
    max_retries: int = 3,
    base_timeout: int = 30,
    backoff_factor: float = 2.0,
    **kwargs
) -> Tuple[bool, Optional[requests.Response], Optional[str]]:
    """
    Make an API call with retry logic and exponential backoff.
    
    Args:
        method: HTTP method ('GET', 'POST', etc.)
        url: API endpoint URL
        max_retries: Maximum number of retry attempts
        base_timeout: Base timeout in seconds
        backoff_factor: Multiplier for exponential backoff
        **kwargs: Additional arguments for requests
        
    Returns:
        Tuple of (success, response, error_message)
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Calculate timeout with exponential backoff
            timeout = base_timeout * (backoff_factor ** attempt)
            
            # Make the request
            if method.upper() == 'GET':
                response = requests.get(url, timeout=timeout, **kwargs)
            elif method.upper() == 'POST':
                response = requests.post(url, timeout=timeout, **kwargs)
            elif method.upper() == 'PUT':
                response = requests.put(url, timeout=timeout, **kwargs)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, timeout=timeout, **kwargs)
            else:
                return False, None, f"Unsupported HTTP method: {method}"
            
            # Check if the response is successful
            if response.status_code < 500:  # Don't retry on client errors (4xx)
                return True, response, None
            
            # Server error (5xx) - retry
            last_error = f"Server error {response.status_code}: {response.text}"
            
        except requests.exceptions.Timeout:
            last_error = f"Request timed out after {timeout} seconds"
        except requests.exceptions.ConnectionError:
            last_error = "Connection error - server may be unreachable"
        except requests.exceptions.RequestException as e:
            last_error = f"Request failed: {str(e)}"
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
        
        # If this isn't the last attempt, wait before retrying
        if attempt < max_retries:
            wait_time = backoff_factor ** attempt
            st.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    
    return False, None, last_error


def make_upload_request_with_retry(
    url: str,
    files: Dict[str, Any],
    data: Dict[str, Any],
    max_retries: int = 2,
    base_timeout: int = 120
) -> Tuple[bool, Optional[requests.Response], Optional[str]]:
    """
    Make a file upload request with retry logic.
    
    Args:
        url: Upload endpoint URL
        files: Files to upload
        data: Form data
        max_retries: Maximum number of retry attempts
        base_timeout: Base timeout in seconds
        
    Returns:
        Tuple of (success, response, error_message)
    """
    return make_api_call_with_retry(
        'POST', url, max_retries, base_timeout, 
        files=files, data=data
    )


def make_chat_request_with_retry(
    url: str,
    json_data: Dict[str, Any],
    max_retries: int = 2,
    base_timeout: int = 90
) -> Tuple[bool, Optional[requests.Response], Optional[str]]:
    """
    Make a chat/translation request with retry logic.
    
    Args:
        url: API endpoint URL
        json_data: JSON data to send
        max_retries: Maximum number of retry attempts
        base_timeout: Base timeout in seconds
        
    Returns:
        Tuple of (success, response, error_message)
    """
    return make_api_call_with_retry(
        'POST', url, max_retries, base_timeout,
        json=json_data
    )


def make_get_request_with_retry(
    url: str,
    max_retries: int = 2,
    base_timeout: int = 30,
    **kwargs
) -> Tuple[bool, Optional[requests.Response], Optional[str]]:
    """
    Make a GET request with retry logic.
    
    Args:
        url: API endpoint URL
        max_retries: Maximum number of retry attempts
        base_timeout: Base timeout in seconds
        **kwargs: Additional arguments for requests
        
    Returns:
        Tuple of (success, response, error_message)
    """
    return make_api_call_with_retry(
        'GET', url, max_retries, base_timeout, **kwargs
    )
