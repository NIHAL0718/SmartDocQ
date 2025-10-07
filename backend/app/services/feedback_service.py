"""Service for handling user feedback and error reports."""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import os
from pathlib import Path

from ..core.config import settings
from ..core.logging import get_logger
from ..core.errors import FeedbackValidationError
from ..models.feedback import (
    FeedbackRequest, 
    FeedbackResponse, 
    ErrorReportRequest, 
    ErrorReportResponse,
    FeedbackSummary,
    FeedbackType,
    IssueType
)

# Initialize logger
logger = get_logger("feedback_service")

# Define paths for storing feedback data
FEEDBACK_DIR = Path(settings.DOCUMENT_STORE_PATH) / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


class FeedbackService:
    """Service for handling user feedback and error reports."""
    
    @staticmethod
    def submit_feedback(feedback: FeedbackRequest) -> FeedbackResponse:
        """Submit feedback for a document or question.
        
        Args:
            feedback (FeedbackRequest): Feedback request model
            
        Returns:
            FeedbackResponse: Feedback response model
            
        Raises:
            FeedbackValidationError: If feedback validation fails
        """
        # Validate feedback
        FeedbackService._validate_feedback(feedback)
        
        # Generate feedback ID
        feedback_id = f"feedback-{uuid.uuid4()}"
        
        # Create feedback data
        feedback_data = {
            "id": feedback_id,
            "document_id": feedback.document_id,
            "question_id": feedback.question_id,
            "feedback_type": feedback.feedback_type,
            "rating": feedback.rating,
            "comment": feedback.comment,
            "issue_type": feedback.issue_type,
            "issue_details": feedback.issue_details,
            "created_at": datetime.now().isoformat()
        }
        
        # Save feedback to file
        FeedbackService._save_feedback(feedback_data)
        
        # Log feedback submission
        logger.info(f"Feedback submitted: {feedback_id} for document {feedback.document_id}")
        
        # Return feedback response
        return FeedbackResponse(**feedback_data)
    
    @staticmethod
    def report_error(report: ErrorReportRequest) -> ErrorReportResponse:
        """Report an error in an answer.
        
        Args:
            report (ErrorReportRequest): Error report request model
            
        Returns:
            ErrorReportResponse: Error report response model
        """
        # Generate report ID
        report_id = f"error-{uuid.uuid4()}"
        
        # Create report data
        report_data = {
            "id": report_id,
            "document_id": report.document_id,
            "question_id": report.question_id,
            "issue_type": report.issue_type,
            "issue_details": report.issue_details,
            "suggested_correction": report.suggested_correction,
            "created_at": datetime.now().isoformat(),
            "status": "submitted"
        }
        
        # Save report to file
        FeedbackService._save_error_report(report_data)
        
        # Log error report submission
        logger.info(f"Error report submitted: {report_id} for question {report.question_id}")
        
        # Return report response
        return ErrorReportResponse(**report_data)
    
    @staticmethod
    def get_feedback_summary(document_id: str) -> FeedbackSummary:
        """Get feedback summary for a document.
        
        Args:
            document_id (str): Document ID
            
        Returns:
            FeedbackSummary: Feedback summary model
        """
        # Get all feedback for the document
        feedback_list = FeedbackService._get_document_feedback(document_id)
        
        # Count feedback by type
        positive_count = sum(1 for f in feedback_list if f.get("feedback_type") == FeedbackType.POSITIVE)
        negative_count = sum(1 for f in feedback_list if f.get("feedback_type") == FeedbackType.NEGATIVE)
        neutral_count = sum(1 for f in feedback_list if f.get("feedback_type") == FeedbackType.NEUTRAL)
        
        # Calculate average rating
        ratings = [f.get("rating") for f in feedback_list if f.get("rating") is not None]
        average_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Count common issues
        issues = [f.get("issue_type") for f in feedback_list if f.get("issue_type") is not None]
        issue_counts = {}
        for issue in issues:
            if issue in issue_counts:
                issue_counts[issue] += 1
            else:
                issue_counts[issue] = 1
        
        common_issues = [{
            "type": issue_type,
            "count": count
        } for issue_type, count in issue_counts.items()]
        
        # Sort common issues by count (descending)
        common_issues.sort(key=lambda x: x["count"], reverse=True)
        
        # Return feedback summary
        return FeedbackSummary(
            total_feedback=len(feedback_list),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            average_rating=average_rating,
            common_issues=common_issues if common_issues else None
        )
    
    @staticmethod
    def get_question_feedback(document_id: str, question_id: str) -> FeedbackSummary:
        """Get feedback summary for a specific question.
        
        Args:
            document_id (str): Document ID
            question_id (str): Question ID
            
        Returns:
            FeedbackSummary: Feedback summary model
        """
        # Get all feedback for the document
        all_feedback = FeedbackService._get_document_feedback(document_id)
        
        # Filter feedback for the specific question
        question_feedback = [f for f in all_feedback if f.get("question_id") == question_id]
        
        # Count feedback by type
        positive_count = sum(1 for f in question_feedback if f.get("feedback_type") == FeedbackType.POSITIVE)
        negative_count = sum(1 for f in question_feedback if f.get("feedback_type") == FeedbackType.NEGATIVE)
        neutral_count = sum(1 for f in question_feedback if f.get("feedback_type") == FeedbackType.NEUTRAL)
        
        # Calculate average rating
        ratings = [f.get("rating") for f in question_feedback if f.get("rating") is not None]
        average_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Count common issues
        issues = [f.get("issue_type") for f in question_feedback if f.get("issue_type") is not None]
        issue_counts = {}
        for issue in issues:
            if issue in issue_counts:
                issue_counts[issue] += 1
            else:
                issue_counts[issue] = 1
        
        common_issues = [{
            "type": issue_type,
            "count": count
        } for issue_type, count in issue_counts.items()]
        
        # Sort common issues by count (descending)
        common_issues.sort(key=lambda x: x["count"], reverse=True)
        
        # Return feedback summary
        return FeedbackSummary(
            total_feedback=len(question_feedback),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            average_rating=average_rating,
            common_issues=common_issues if common_issues else None
        )
    
    @staticmethod
    def _validate_feedback(feedback: FeedbackRequest) -> None:
        """Validate feedback request.
        
        Args:
            feedback (FeedbackRequest): Feedback request model
            
        Raises:
            FeedbackValidationError: If feedback validation fails
        """
        # Check if rating is provided for positive/negative feedback
        if feedback.feedback_type in [FeedbackType.POSITIVE, FeedbackType.NEGATIVE] and feedback.rating is None:
            raise FeedbackValidationError("Rating is required for positive or negative feedback")
        
        # Check if issue type and details are provided for negative feedback
        if feedback.feedback_type == FeedbackType.NEGATIVE and (feedback.issue_type is None or feedback.issue_details is None):
            raise FeedbackValidationError("Issue type and details are required for negative feedback")
    
    @staticmethod
    def _save_feedback(feedback_data: Dict[str, Any]) -> None:
        """Save feedback data to file.
        
        Args:
            feedback_data (Dict[str, Any]): Feedback data
        """
        # Create document feedback directory if it doesn't exist
        document_feedback_dir = FEEDBACK_DIR / feedback_data["document_id"]
        document_feedback_dir.mkdir(exist_ok=True)
        
        # Save feedback to file
        feedback_file = document_feedback_dir / f"{feedback_data['id']}.json"
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)
    
    @staticmethod
    def _save_error_report(report_data: Dict[str, Any]) -> None:
        """Save error report data to file.
        
        Args:
            report_data (Dict[str, Any]): Error report data
        """
        # Create document feedback directory if it doesn't exist
        document_feedback_dir = FEEDBACK_DIR / report_data["document_id"]
        document_feedback_dir.mkdir(exist_ok=True)
        
        # Save report to file
        report_file = document_feedback_dir / f"{report_data['id']}.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
    
    @staticmethod
    def _get_document_feedback(document_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a document.
        
        Args:
            document_id (str): Document ID
            
        Returns:
            List[Dict[str, Any]]: List of feedback data
        """
        # Check if document feedback directory exists
        document_feedback_dir = FEEDBACK_DIR / document_id
        if not document_feedback_dir.exists():
            return []
        
        # Get all feedback files
        feedback_files = list(document_feedback_dir.glob("*.json"))
        
        # Read feedback data from files
        feedback_list = []
        for file in feedback_files:
            try:
                with open(file, "r") as f:
                    feedback_data = json.load(f)
                    feedback_list.append(feedback_data)
            except Exception as e:
                logger.error(f"Error reading feedback file {file}: {e}")
        
        return feedback_list