"""API endpoints for OCR (Optical Character Recognition) functionality."""

import os
import uuid
import json
import glob
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from app.services.ocr_service import process_image_ocr

router = APIRouter()

# Directory to store uploaded images
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", "ocr")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class OcrResponse(BaseModel):
    """Response model for OCR processing."""
    id: str
    status: str
    message: str
    text: Optional[str] = None
    confidence: Optional[float] = None


class OcrResult(BaseModel):
    """Model for OCR result."""
    id: str
    status: str
    text: Optional[str] = None
    confidence: Optional[float] = None
    language: Optional[str] = None  # Detected language
    processing_time: Optional[float] = None  # in seconds
    error: Optional[str] = None


@router.post("/process", response_model=OcrResponse)
async def process_ocr(
    file: UploadFile = File(...),
    enhance_image: bool = Form(False),
):
    """Process an image with OCR.
    
    The language will be automatically detected.
    """
    # Generate a unique ID for this OCR job
    ocr_id = str(uuid.uuid4())
    
    # Create the uploads directory if it doesn't exist
    os.makedirs("uploads/ocr", exist_ok=True)
    
    # Save the uploaded file
    file_path = f"uploads/ocr/{ocr_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process the image with OCR
    # We'll process it directly instead of in the background for simplicity
    ocr_result = process_image_ocr(ocr_id, file_path, None, enhance_image)  # None for language to trigger auto-detection
    
    if ocr_result["status"] == "error":
        return {
            "id": ocr_id,
            "status": "error",
            "message": f"OCR processing failed: {ocr_result.get('error', 'Unknown error')}",
        }
    
    return {
        "id": ocr_id,
        "status": "success",
        "message": "OCR processing completed. Language was automatically detected.",
        "text": ocr_result.get("text", ""),
        "confidence": ocr_result.get("confidence", 0.0),
    }


@router.get("/status/{ocr_id}", response_model=OcrResponse)
async def get_ocr_status(ocr_id: str):
    """Get the processing status of an OCR job."""
    try:
        # Check if the OCR result file exists
        result_file = f"uploads/ocr/{ocr_id}_result.json"
        
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    result = json.load(f)
                
                if result.get("status") == "error":
                    return OcrResponse(
                        id=ocr_id,
                        status="error",
                        message=f"OCR processing failed: {result.get('error', 'Unknown error')}",
                    )
                
                return OcrResponse(
                    id=ocr_id,
                    status="completed",
                    message="OCR processing completed.",
                    text=result.get("text", ""),
                    confidence=result.get("confidence", 0.0),
                )
            except Exception as e:
                return OcrResponse(
                    id=ocr_id,
                    status="error",
                    message=f"Error reading OCR result: {str(e)}",
                )
        
        # Check if the file exists but processing hasn't completed
        import glob
        file_pattern = f"uploads/ocr/{ocr_id}_*"
        matching_files = glob.glob(file_pattern)
        
        if matching_files:
            return OcrResponse(
                id=ocr_id,
                status="processing",
                message="OCR processing is still in progress.",
            )
        
        return OcrResponse(
            id=ocr_id,
            status="error",
            message="OCR job not found.",
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{ocr_id}", response_model=OcrResult)
async def get_ocr_result(ocr_id: str):
    """Get the result of a completed OCR job."""
    try:
        # Check if the OCR result file exists
        result_file = f"uploads/ocr/{ocr_id}_result.json"
        
        if not os.path.exists(result_file):
            # Check if the file exists but processing hasn't completed
            import glob
            file_pattern = f"uploads/ocr/{ocr_id}_*"
            matching_files = glob.glob(file_pattern)
            
            if matching_files:
                return OcrResult(
                    id=ocr_id,
                    status="processing",
                    error="OCR processing is still in progress.",
                )
            
            return OcrResult(
                id=ocr_id,
                status="error",
                error="OCR job not found.",
            )
        
        try:
            import json
            with open(result_file, "r") as f:
                result = json.load(f)
            
            if result.get("status") == "error":
                return OcrResult(
                    id=ocr_id,
                    status="error",
                    error=f"OCR processing failed: {result.get('error', 'Unknown error')}",
                )
            
            return OcrResult(
                id=ocr_id,
                status="completed",
                text=result.get("text", ""),
                confidence=result.get("confidence", 0.0),
                language=result.get("language", "english"),
                processing_time=result.get("processing_time", 0.0),
            )
        except Exception as e:
            return OcrResult(
                id=ocr_id,
                status="error",
                error=f"Error reading OCR result: {str(e)}",
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/to-document/{ocr_id}")
async def convert_ocr_to_document(ocr_id: str):
    """Convert OCR result to a searchable document."""
    try:
        # In a real implementation, this would:
        # 1. Retrieve the OCR result
        # 2. Create a new document with the extracted text
        # 3. Process it like a regular document (chunking, embedding, etc.)
        # For now, we'll return a mock response
        return {
            "ocr_id": ocr_id,
            "doc_id": f"doc-from-ocr-{ocr_id}",
            "status": "processing",
            "message": "OCR result is being converted to a searchable document.",
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))