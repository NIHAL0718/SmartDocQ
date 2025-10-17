"""Models for feedback functionality."""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class FeedbackType(str, Enum):
    """Types of feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class IssueType(str, Enum):
    """Types of issues that can be reported."""
    INCORRECT_INFORMATION = "incorrect_information"
    MISSING_INFORMATION = "missing_information"
    IRRELEVANT_ANSWER = "irrelevant_answer"
    HALLUCINATION = "hallucination"
    HARMFUL_CONTENT = "harmful_content"
    OTHER = "other"


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    document_id: str
    question_id: Optional[str] = None
    feedback_type: FeedbackType
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = None
    issue_type: Optional[IssueType] = None
    issue_details: Optional[str] = None
    
    @validator("rating")
    def validate_rating(cls, v, values):
        """Validate that rating is provided for positive/negative feedback."""
        if values.get("feedback_type") in [FeedbackType.POSITIVE, FeedbackType.NEGATIVE] and v is None:
            raise ValueError("Rating is required for positive or negative feedback")
        return v
    
    @validator("issue_type", "issue_details")
    def validate_issue_fields(cls, v, values):
        """Validate that issue fields are provided when reporting an issue."""
        if values.get("feedback_type") == FeedbackType.NEGATIVE and v is None:
            raise ValueError("Issue type and details are required for negative feedback")
        return v


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    id: str
    document_id: str
    question_id: Optional[str] = None
    feedback_type: FeedbackType
    rating: Optional[int] = None
    comment: Optional[str] = None
    issue_type: Optional[IssueType] = None
    issue_details: Optional[str] = None
    created_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "id": "feedback-123",
                "document_id": "doc-456",
                "question_id": "q-789",
                "feedback_type": "positive",
                "rating": 5,
                "comment": "The answer was very helpful and accurate.",
                "issue_type": None,
                "issue_details": None,
                "created_at": "2023-06-15T10:30:00"
            }
        }


class ErrorReportRequest(BaseModel):
    """Request model for reporting an error in an answer."""
    document_id: str
    question_id: str
    issue_type: IssueType
    issue_details: str
    suggested_correction: Optional[str] = None


class ErrorReportResponse(BaseModel):
    """Response model for error report submission."""
    id: str
    document_id: str
    question_id: str
    issue_type: IssueType
    issue_details: str
    suggested_correction: Optional[str] = None
    created_at: datetime
    status: str = "submitted"


class FeedbackSummary(BaseModel):
    """Summary of feedback for a document or question."""
    total_feedback: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_rating: Optional[float] = None
    common_issues: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "total_feedback": 25,
                "positive_count": 18,
                "negative_count": 5,
                "neutral_count": 2,
                "average_rating": 4.2,
                "common_issues": [
                    {"type": "missing_information", "count": 3},
                    {"type": "incorrect_information", "count": 2}
                ]
            }
        }