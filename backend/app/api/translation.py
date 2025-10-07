"""API endpoints for translation functionality."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List

from app.models.qa import TranslationRequest, TranslationResponse
from app.services.translation_service import TranslationService
from app.core.errors import TranslationError, http_exception_handler
from app.core.dependencies import get_translation_service

# Create router
router = APIRouter()


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
        result = await translation_service.translate_text(
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language
        )
        
        # Check if there was an error in the translation
        if "error" in result:
            # We still return the result with the error message included
            # The frontend will handle displaying the error appropriately
            pass
        
        return result
    except TranslationError as e:
        raise http_exception_handler(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during translation: {str(e)}"
        )


@router.get("/languages")
def get_supported_languages(
    translation_service: TranslationService = Depends(get_translation_service)
):
    """Get a list of supported languages for translation.
    
    Args:
        translation_service: Translation service dependency
        
    Returns:
        List: List of dictionaries with language codes and names
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
    request: dict,
    translation_service: TranslationService = Depends(get_translation_service)
) -> Dict[str, any]:
    """Detect the language of a text.
    
    Args:
        request: Dictionary containing text to detect language for
        translation_service: Translation service dependency
        
    Returns:
        Dict: Dictionary with detected language code and confidence
    """
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text is required for language detection"
            )
            
        language_code = await translation_service.detect_language(text)
        return {"detected_language": language_code, "confidence": 0.9}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect language: {str(e)}"
        )