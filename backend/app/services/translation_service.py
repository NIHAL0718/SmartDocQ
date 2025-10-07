"""Translation service for the SmartDocQ application."""

import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

from ..core.logging import get_logger
from ..core.config import settings
from ..core.errors import TranslationError
from ..models.qa import TranslationRequest, TranslationResponse
from ..services.qa_service import translate_text as qa_translate_text

# Initialize logger
logger = get_logger("translation_service")


class TranslationService:
    """Service for text translation."""
    
    def __init__(self, llm_service=None):
        """Initialize the translation service.
        
        Args:
            llm_service: Service for language model interactions (not used directly, kept for compatibility)
        """
        # We'll use the qa_service's translate_text function directly
        pass
    
    async def translate_text(self, translation_request: TranslationRequest) -> TranslationResponse:
        """Translate text to the target language.
        
        Args:
            translation_request: Translation request object
            
        Returns:
            TranslationResponse: Translation response object
            
        Raises:
            TranslationError: If there is an error translating the text
        """
        start_time = time.time()
        
        try:
            # Get the text to translate
            text = translation_request.text
            
            # Get the target language
            target_language = translation_request.target_language
            
            # Get the source language if provided
            source_language = translation_request.source_language
            
            # If source language is not provided, detect it
            if not source_language:
                source_language = await self.detect_language(text)
            
            # Translate the text using the qa_service's translate_text function
            # Note: qa_service's translate_text is not async, so we don't need to await it
            translation_result = qa_translate_text(
                text=text,
                target_language=target_language,
                source_language=source_language
            )
            
            # Extract the translated text from the result
            if isinstance(translation_result, dict):
                translated_text = translation_result.get("translated_text", text)
            else:
                translated_text = translation_result
            
            # Create response
            response = TranslationResponse(
                original_text=text,
                translated_text=translated_text,
                target_language=target_language,
                source_language=source_language,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            raise TranslationError(detail=f"Error translating text: {e}", 
                                  target_language=translation_request.target_language,
                                  source_language=translation_request.source_language)
    
    # The _perform_translation method has been removed as we're now using
    # the qa_service's translate_text function directly
    
    async def text_to_text_translate(self, translation_request: TextToTextTranslationRequest) -> TranslationResponse:
        """Translate text to the target language using text-to-text approach.
        
        This method provides an alternative translation method that may work better
        for certain language pairs or text types.
        
        Args:
            translation_request: Text-to-text translation request object
            
        Returns:
            TranslationResponse: Translation response object
            
        Raises:
            TranslationError: If there is an error translating the text
        """
        start_time = time.time()
        
        try:
            # Get the text to translate
            text = translation_request.text
            
            # Get the target language
            target_language = translation_request.target_language
            
            # Get the source language if provided
            source_language = translation_request.source_language
            
            # If source language is not provided, detect it
            if not source_language:
                source_language = await self.detect_language(text)
            
            # Use a more direct text-to-text approach for translation
            # This could be implemented with a different model or approach
            # For now, we'll use the same qa_translate_text function but with a flag
            # indicating it's a text-to-text request
            translation_result = qa_translate_text(
                text=text,
                target_language=target_language,
                source_language=source_language,
                use_text_to_text=True  # Flag to indicate text-to-text translation
            )
            
            # Extract the translated text from the result
            if isinstance(translation_result, dict):
                translated_text = translation_result.get("translated_text", text)
            else:
                translated_text = translation_result
            
            # Create response
            response = TranslationResponse(
                original_text=text,
                translated_text=translated_text,
                target_language=target_language,
                source_language=source_language,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error in text-to-text translation: {e}")
            raise TranslationError(detail=f"Error in text-to-text translation: {e}", 
                                  target_language=translation_request.target_language,
                                  source_language=translation_request.source_language)
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of a text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            str: Detected language code
        """
        # This is a simplified implementation that could be enhanced with a proper language detection library
        # For now, we'll use a simple approach based on common language patterns
        
        try:
            # Import langdetect if available
            try:
                from langdetect import detect
                language_code = detect(text)
                return language_code
            except ImportError:
                # Fallback to a very simple detection based on character sets
                # This is not accurate but serves as a basic fallback
                text = text.lower()
                
                # Check for common scripts and patterns
                # Chinese characters
                if any('\u4e00' <= char <= '\u9fff' for char in text):
                    return "zh"
                # Japanese specific characters (hiragana and katakana)
                elif any('\u3040' <= char <= '\u30ff' for char in text):
                    return "ja"
                # Korean specific characters (Hangul)
                elif any('\uac00' <= char <= '\ud7a3' for char in text):
                    return "ko"
                # Arabic script
                elif any('\u0600' <= char <= '\u06ff' for char in text):
                    return "ar"
                # Cyrillic script (Russian, etc.)
                elif any('\u0400' <= char <= '\u04ff' for char in text):
                    return "ru"
                # Devanagari script (Hindi, etc.)
                elif any('\u0900' <= char <= '\u097f' for char in text):
                    return "hi"
                # Thai script
                elif any('\u0e00' <= char <= '\u0e7f' for char in text):
                    return "th"
                
                # Default to English for Latin script
                return "en"
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            # Return 'en' as fallback
            return "en"
    
    def get_supported_languages(self) -> list:
        """Get a list of supported languages for translation.
        
        Returns:
            list: List of dictionaries with language codes and names
        """
        # Return a list of supported languages in the format expected by the frontend
        languages = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "nl", "name": "Dutch"},
            {"code": "ru", "name": "Russian"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "bn", "name": "Bengali"},
            {"code": "pa", "name": "Punjabi"},
            {"code": "te", "name": "Telugu"},
            {"code": "ta", "name": "Tamil"},
            {"code": "ur", "name": "Urdu"},
            {"code": "th", "name": "Thai"},
            {"code": "vi", "name": "Vietnamese"},
        ]
        return languages