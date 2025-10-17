"""API endpoints for user feedback on answers."""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

router = APIRouter()


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    question_id: str
    rating: int  # 1-5 rating
    comment: Optional[str] = None
    is_error_report: bool = False
    error_type: Optional[str] = None  # e.g., "incorrect_answer", "missing_context", etc.


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    id: str
    status: str
    message: str


class FeedbackSummary(BaseModel):
    """Summary of feedback for a document or question."""
    average_rating: float
    total_ratings: int
    error_reports: int
    recent_comments: List[str]


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for an answer."""
    try:
        # In a real implementation, this would save the feedback to a database
        # For now, we'll return a mock response
        
        # Validate rating
        if feedback.rating < 1 or feedback.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Process error report if applicable
        if feedback.is_error_report and not feedback.error_type:
            raise HTTPException(status_code=400, detail="Error type is required for error reports")
        
        return FeedbackResponse(
            id="feedback-123",  # This would be a real ID in production
            status="success",
            message="Feedback submitted successfully",
        )
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/{doc_id}", response_model=FeedbackSummary)
async def get_feedback_summary(doc_id: str):
    """Get summary of feedback for a document."""
    try:
        # In a real implementation, this would query the database for feedback
        # For now, we'll return a mock response
        return FeedbackSummary(
            average_rating=4.2,
            total_ratings=15,
            error_reports=2,
            recent_comments=[
                "Very helpful answer!",
                "The answer was mostly correct but missed some details.",
                "Great explanation, thank you!",
            ],
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/question/{question_id}", response_model=FeedbackSummary)
async def get_question_feedback(question_id: str):
    """Get feedback for a specific question."""
    try:
        # In a real implementation, this would query the database for feedback
        # For now, we'll return a mock response
        return FeedbackSummary(
            average_rating=4.0,
            total_ratings=3,
            error_reports=0,
            recent_comments=[
                "This answer was exactly what I needed!",
            ],
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report-error")
async def report_error(
    question_id: str = Body(...),
    error_type: str = Body(...),
    description: str = Body(...),
):
    """Report an error in an answer."""
    try:
        # In a real implementation, this would save the error report to a database
        # For now, we'll return a mock response
        return {
            "status": "success",
            "message": "Error report submitted successfully",
            "report_id": "error-report-123",  # This would be a real ID in production
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))