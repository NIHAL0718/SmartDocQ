"""Utility functions for translation functionality in the frontend."""

import requests
import streamlit as st
import os
import time
import unicodedata

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000/api")


@st.cache_data(ttl=30)
def check_backend_connection():
    """Check if the backend server is running and accessible.
    
    Returns:
        bool: True if backend is accessible, False otherwise
    """
    try:
        # Try to connect to the health endpoint
        response = requests.get(f"{API_URL.split('/api')[0]}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def translate_text(text, target_language, source_language=None):
    """Translate text to the target language.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code
        source_language (str, optional): Source language code. If None, it will be auto-detected.
        
    Returns:
        dict: Translation response with translated text and detected source language
    """
    # First check if backend is accessible
    if not check_backend_connection():
        st.error("⚠️ Backend server is not accessible. Please check if the server is running.")
        return {"translated_text": text, "source_language": source_language or "unknown", "error": "Backend connection failed"}
    
    try:
        # Prepare request data
        data = {
            "text": text,
            "target_language": target_language
        }
        
        # Add source language if provided
        if source_language:
            data["source_language"] = source_language
        
        # Make API request with timeout
        response = requests.post(
            f"{API_URL}/translation/translate",
            json=data,
            timeout=10  # Add timeout to prevent hanging
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Get translation data
        translation_data = response.json()
        
        # Check if there's an error in the translation data
        if "error" in translation_data:
            error_message = translation_data["error"]
            # Only show error if it's not related to fallback translation
            if not "fallback translation" in error_message.lower():
                st.error(f"⚠️ Translation error: {error_message}")
            # Return the data with the error so the calling function can handle it appropriately
            return translation_data
            
        # Remove any note text that might be in the translated text
        if "translated_text" in translation_data:
            translated_text = translation_data["translated_text"]
            # Remove the note if present
            if "\n\n[Note:" in translated_text:
                translation_data["translated_text"] = translated_text.split("\n\n[Note:")[0]
        
        # Error correction for common translation issues
        if "translated_text" in translation_data:
            translated_text = translation_data["translated_text"]
            
            # Fix common translation errors
            # 1. Fix missing spaces after punctuation
            import re
            translated_text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', translated_text)
            
            # 2. Fix repeated words
            translated_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', translated_text)
            
            # 3. Fix capitalization after sentence-ending punctuation
            translated_text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), translated_text)
            
            # 4. Fix encoding issues for Spanish characters
            if target_language == "es":
                # Fix common Spanish encoding issues
                spanish_fixes = {
                    # Common words with encoding issues
                    "traducciÃ³n": "traducción",
                    "tÃº": "tú",
                    "gracias para": "gracias por",
                    "gracias tú para": "gracias por",
                    
                    # Accented vowels
                    "Ã¡": "á",
                    "Ã©": "é",
                    "Ã­": "í",
                    "Ã³": "ó",
                    "Ãº": "ú",
                    
                    # Ñ character
                    "Ã±": "ñ",
                    
                    # Other common Spanish words with accents
                    "informaciÃ³n": "información",
                    "comunicaciÃ³n": "comunicación",
                    "educaciÃ³n": "educación",
                    "situaciÃ³n": "situación",
                    "atenciÃ³n": "atención",
                    "opciÃ³n": "opción",
                    "acciÃ³n": "acción",
                    "relaciÃ³n": "relación",
                    "direcciÃ³n": "dirección",
                    "corazÃ³n": "corazón",
                    "razÃ³n": "razón",
                    "canciÃ³n": "canción",
                    "maÃ±ana": "mañana",
                    "espaÃ±ol": "español",
                    "seÃ±or": "señor",
                    "compaÃ±ero": "compañero",
                    "niÃ±o": "niño",
                    "aÃ±o": "año",
                    "peÃ±a": "peña",
                    "seÃ±al": "señal",
                    "montaÃ±a": "montaña"
                }
                
                # Apply Spanish fixes
                for incorrect, correct in spanish_fixes.items():
                    translated_text = translated_text.replace(incorrect, correct)
                
                # Apply Unicode normalization to ensure consistent character representation
                translated_text = unicodedata.normalize('NFC', translated_text)
            
            # Update the translation data
            translation_data["translated_text"] = translated_text
        
        # Return corrected translation data
        return translation_data
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Connection error: Unable to connect to the translation service. Please check your network connection.")
        return {"translated_text": text, "source_language": source_language or "unknown", "error": "Connection error"}


@st.cache_data(ttl=3600)
def get_supported_languages():
    """Get list of supported languages for translation.
    
    Returns:
        list: List of supported language dictionaries with code and name
    """
    # First check if backend is accessible
    if not check_backend_connection():
        st.warning("⚠️ Backend server is not accessible. Using default language list.")
        return _get_default_languages()
    
    try:
        # Make API request with timeout
        response = requests.get(f"{API_URL}/translation/languages", timeout=5)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        # Handle different response formats
        if isinstance(data, dict) and 'languages' in data:
            # Format from simple_server.py
            return data['languages']
        elif isinstance(data, list):
            # Format from main.py
            return data
        else:
            st.warning("⚠️ Unexpected response format from language API. Using default language list.")
            return _get_default_languages()
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ Error fetching supported languages: {str(e)}. Using default language list.")
        return _get_default_languages()


def _get_default_languages():
    """Get a default list of supported languages when API is not available.
    
    Returns:
        list: Default list of language dictionaries
    """
    return [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "ru", "name": "Russian"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"},
        {"code": "ko", "name": "Korean"},
        {"code": "ar", "name": "Arabic"},
        {"code": "hi", "name": "Hindi"},
        {"code": "bn", "name": "Bengali"},
        {"code": "te", "name": "Telugu"},
        {"code": "ta", "name": "Tamil"}
    ]


def detect_language(text):
    """Detect the language of the given text.
    
    Args:
        text (str): Text to detect language for
        
    Returns:
        dict: Detected language information
    """
    # First check if backend is accessible
    if not check_backend_connection():
        st.warning("⚠️ Backend server is not accessible. Using English as default language.")
        return {"detected_language": "en", "confidence": 0.0, "error": "Backend connection failed"}
    
    try:
        # Make API request with timeout
        response = requests.post(
            f"{API_URL}/translation/detect-language",
            json={"text": text},
            timeout=5  # Add timeout to prevent hanging
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        # Handle different response formats
        if "detected_language" in data:
            # Format from main.py
            return data
        elif "language" in data:
            # Format from simple_server.py
            return {"detected_language": data["language"], "confidence": data.get("confidence", 0.9)}
        else:
            st.warning("⚠️ Unexpected response format from language detection API.")
            return {"detected_language": "en", "confidence": 0.0, "error": "Unexpected response format"}
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Connection error: Unable to connect to the language detection service.")
        return {"detected_language": "en", "confidence": 0.0, "error": "Connection error"}
    except requests.exceptions.Timeout:
        st.warning("⚠️ Request timed out: The language detection service is taking too long to respond.")
        return {"detected_language": "en", "confidence": 0.0, "error": "Timeout"}
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ Error detecting language: {str(e)}")
        return {"detected_language": "en", "confidence": 0.0, "error": str(e)}


def text_to_text_translate(text, target_language, source_language=None):
    """Translate text to the target language using a direct text-to-text approach.
    
    This function provides an alternative translation method that may work better
    for certain language pairs or text types.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code
        source_language (str, optional): Source language code. If None, it will be auto-detected.
        
    Returns:
        dict: Translation response with translated text and detected source language
    """
    # First check if backend is accessible
    if not check_backend_connection():
        st.error("⚠️ Backend server is not accessible. Please check if the server is running.")
        return {"translated_text": text, "source_language": source_language or "unknown", "error": "Backend connection failed"}
    
    try:
        # Prepare request data
        data = {
            "text": text,
            "target_language": target_language,
            "use_text_to_text": True  # Flag to indicate text-to-text translation
        }
        
        # Add source language if provided
        if source_language:
            data["source_language"] = source_language
        
        # Make API request with timeout
        response = requests.post(
            f"{API_URL}/translation/text-to-text",  # New endpoint for text-to-text translation
            json=data,
            timeout=15  # Slightly longer timeout as text-to-text might take more time
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Get translation data
        translation_data = response.json()
        
        # Check if there's an error in the translation data
        if "error" in translation_data:
            error_message = translation_data["error"]
            # Only show error if it's not related to fallback translation
            if not "fallback translation" in error_message.lower():
                st.error(f"⚠️ Translation error: {error_message}")
            # Return the data with the error so the calling function can handle it appropriately
            return translation_data
        
        # Return translation data
        return translation_data
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Connection error: Unable to connect to the translation service. Please check your network connection.")
        return {"translated_text": text, "source_language": source_language or "unknown", "error": "Connection error"}
    except requests.exceptions.Timeout:
        st.error("⚠️ Request timed out: The translation service is taking too long to respond.")
        return {"translated_text": text, "source_language": source_language or "unknown", "error": "Timeout"}
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Error during translation: {str(e)}")
        return {"translated_text": text, "source_language": source_language or "unknown", "error": str(e)}