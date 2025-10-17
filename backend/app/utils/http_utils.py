"""Utility functions for HTTP requests."""

import re
import json
import aiohttp
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse

from ..core.logging import get_logger
from ..core.errors import InvalidURLError, WebPageProcessingError

# Initialize logger
logger = get_logger("http_utils")

# Regular expression for URL validation
URL_REGEX = re.compile(
    r'^(?:http|https)://'
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
    r'localhost|'
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
    r'(?::\d+)?'
    r'(?:/?|[/?]\S+)$', re.IGNORECASE
)


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    return bool(URL_REGEX.match(url))


def normalize_url(url: str) -> str:
    """Normalize a URL by adding http:// if missing.
    
    Args:
        url (str): URL to normalize
        
    Returns:
        str: Normalized URL
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


def get_domain(url: str) -> str:
    """Extract the domain from a URL.
    
    Args:
        url (str): URL to extract domain from
        
    Returns:
        str: Domain name
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc


async def fetch_url(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, Any]]:
    """Fetch content from a URL.
    
    Args:
        url (str): URL to fetch
        headers (Optional[Dict[str, str]]): HTTP headers
        
    Returns:
        Tuple[str, Dict[str, Any]]: Tuple of (content, metadata)
        
    Raises:
        InvalidURLError: If URL is invalid
        WebPageProcessingError: If an error occurs while fetching the URL
    """
    # Normalize and validate URL
    url = normalize_url(url)
    if not is_valid_url(url):
        raise InvalidURLError(url)
    
    # Default headers
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
    
    # Initialize metadata
    metadata = {
        'url': url,
        'domain': get_domain(url),
        'status_code': None,
        'content_type': None,
        'headers': {},
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                # Update metadata
                metadata['status_code'] = response.status
                metadata['content_type'] = response.headers.get('Content-Type', '')
                metadata['headers'] = dict(response.headers)
                
                # Check if response is successful
                if response.status != 200:
                    error_msg = f"Failed to fetch URL: {url} (Status: {response.status})"
                    logger.error(error_msg)
                    raise WebPageProcessingError(error_msg)
                
                # Get content
                content = await response.text()
                return content, metadata
    except aiohttp.ClientError as e:
        error_msg = f"Error fetching URL: {url} - {str(e)}"
        logger.error(error_msg)
        raise WebPageProcessingError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error fetching URL: {url} - {str(e)}"
        logger.error(error_msg)
        raise WebPageProcessingError(error_msg)


async def post_json(url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Send a POST request with JSON data.
    
    Args:
        url (str): URL to send request to
        data (Dict[str, Any]): JSON data to send
        headers (Optional[Dict[str, str]]): HTTP headers
        
    Returns:
        Dict[str, Any]: Response data
        
    Raises:
        InvalidURLError: If URL is invalid
        WebPageProcessingError: If an error occurs while sending the request
    """
    # Normalize and validate URL
    url = normalize_url(url)
    if not is_valid_url(url):
        raise InvalidURLError(url)
    
    # Default headers
    if headers is None:
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
    else:
        # Ensure Content-Type is set
        headers['Content-Type'] = headers.get('Content-Type', 'application/json')
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers, timeout=30) as response:
                # Check if response is successful
                if response.status != 200:
                    error_msg = f"Failed to post to URL: {url} (Status: {response.status})"
                    logger.error(error_msg)
                    raise WebPageProcessingError(error_msg)
                
                # Get response data
                response_data = await response.json()
                return response_data
    except aiohttp.ClientError as e:
        error_msg = f"Error posting to URL: {url} - {str(e)}"
        logger.error(error_msg)
        raise WebPageProcessingError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding JSON response from URL: {url} - {str(e)}"
        logger.error(error_msg)
        raise WebPageProcessingError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error posting to URL: {url} - {str(e)}"
        logger.error(error_msg)
        raise WebPageProcessingError(error_msg)


async def download_file(url: str, output_path: str, headers: Optional[Dict[str, str]] = None) -> str:
    """Download a file from a URL.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save the file to
        headers (Optional[Dict[str, str]]): HTTP headers
        
    Returns:
        str: Path to the downloaded file
        
    Raises:
        InvalidURLError: If URL is invalid
        WebPageProcessingError: If an error occurs while downloading the file
    """
    # Normalize and validate URL
    url = normalize_url(url)
    if not is_valid_url(url):
        raise InvalidURLError(url)
    
    # Default headers
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=60) as response:
                # Check if response is successful
                if response.status != 200:
                    error_msg = f"Failed to download file from URL: {url} (Status: {response.status})"
                    logger.error(error_msg)
                    raise WebPageProcessingError(error_msg)
                
                # Save the file
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024 * 1024)  # 1 MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                
                logger.info(f"File downloaded from {url} to {output_path}")
                return output_path
    except aiohttp.ClientError as e:
        error_msg = f"Error downloading file from URL: {url} - {str(e)}"
        logger.error(error_msg)
        raise WebPageProcessingError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error downloading file from URL: {url} - {str(e)}"
        logger.error(error_msg)
        raise WebPageProcessingError(error_msg)


def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text.
    
    Args:
        text (str): Text to extract URLs from
        
    Returns:
        List[str]: List of URLs
    """
    # Regular expression for URL extraction
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # Find all URLs in the text
    urls = url_pattern.findall(text)
    
    # Filter out invalid URLs
    valid_urls = [url for url in urls if is_valid_url(url)]
    
    return valid_urls