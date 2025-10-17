"""Main FastAPI application entry point for SmartDocQ backend."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import API routers
from app.api.documents import router as documents_router
from app.api.questions import router as questions_router
from app.api.feedback import router as feedback_router
from app.api.chat import router as chat_router
from app.api.ocr import router as ocr_router
from app.api.translation import router as translation_router
# Add auth routes
from app.routes.auth import router as auth_router

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SmartDocQ API",
    description="AI-powered Document Question Answering API",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins for now
        "https://smart-doc-4o5kgx5j4-nihal-chandras-projects.vercel.app",
        "https://smart-doc.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])
app.include_router(questions_router, prefix="/api/questions", tags=["Questions"])
app.include_router(feedback_router, prefix="/api/feedback", tags=["Feedback"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(ocr_router, prefix="/api/ocr", tags=["OCR"])
app.include_router(translation_router, prefix="/api/translation", tags=["Translation"])
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint to check if API is running."""
    return {"message": "Welcome to SmartDocQ API", "status": "online"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)