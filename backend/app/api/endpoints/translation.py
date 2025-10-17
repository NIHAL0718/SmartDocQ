"""API endpoints for translation functionality."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List

from ...models.qa import TranslationRequest, TranslationResponse, TextToTextTranslationRequest
from ...services.translation_service import TranslationService
from ...core.errors import TranslationError, http_exception_handler
from ...core.dependencies import get_translation_service

# Create router
router = APIRouter(prefix="/translation", tags=["translation"])


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    translation_service: TranslationService = Depends(get_translation_service)
) -> TranslationResponse:
    """Translate text to the target language.
    
    Args:
        request: Translation request with text and target language
        translation_service: Translation service dependency
        
    Returns:
        TranslationResponse: Translation response with translated text
        
    Raises:
        HTTPException: If translation fails
    """
    try:
        response = await translation_service.translate_text(request)
        return response
    except TranslationError as e:
        raise http_exception_handler(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during translation: {str(e)}"
        )


@router.get("/languages", response_model=Dict[str, str])
def get_supported_languages(
    translation_service: TranslationService = Depends(get_translation_service)
) -> Dict[str, str]:
    """Get a list of supported languages for translation.
    
    Args:
        translation_service: Translation service dependency
        
    Returns:
        Dict[str, str]: Dictionary of language codes and names
    """
    try:
        languages = translation_service.get_supported_languages()
        return languages
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve supported languages: {str(e)}"
        )


@router.post("/detect-language")
async def detect_language(
    text: str,
    translation_service: TranslationService = Depends(get_translation_service)
) -> Dict[str, str]:
    """Detect the language of a text.
    
    Args:
        text: Text to detect language for
        translation_service: Translation service dependency
        
    Returns:
        Dict[str, str]: Dictionary with detected language code
    """
    try:
        language_code = await translation_service.detect_language(text)
        return {"language": language_code}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect language: {str(e)}"
        )


@router.post("/text-to-text", response_model=TranslationResponse)
async def text_to_text_translate(
    request: TextToTextTranslationRequest,
    translation_service: TranslationService = Depends(get_translation_service)
) -> TranslationResponse:
    """Translate text to the target language using text-to-text approach.
    
    This endpoint provides an alternative translation method that may work better
    for certain language pairs or text types.
    
    Args:
        request: Text-to-text translation request with text and target language
        translation_service: Translation service dependency
        
    Returns:
        TranslationResponse: Translation response with translated text
        
    Raises:
        HTTPException: If translation fails
    """
    try:
        response = await translation_service.text_to_text_translate(request)
        return response
    except TranslationError as e:
        raise http_exception_handler(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during text-to-text translation: {str(e)}"
        )