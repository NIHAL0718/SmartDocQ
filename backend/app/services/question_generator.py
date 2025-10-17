"""Service for generating important questions from documents."""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

from app.models.document import DocumentChunk

# Load environment variables
load_dotenv()

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Set up the model
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-pro")


def generate_important_questions(doc_chunks: List[DocumentChunk], count: int = 5) -> List[str]:
    try:
        if not GEMINI_API_KEY:
            return ["API key not configured. Please set GEMINI_API_KEY environment variable."]
        
        max_chunks = min(len(doc_chunks), 10)
        combined_text = "\n\n---\n\n".join([chunk.text for chunk in doc_chunks[:max_chunks]])

        prompt = f"""
        ### Document Content:
        {combined_text}

        ### Instructions:
        Based on the document content provided above, generate {count} important and insightful questions that would help someone understand the key points, concepts, and implications of this document.

        The questions should:
        1. Cover the most important information in the document
        2. Be diverse and cover different aspects of the content
        3. Range from factual to analytical/interpretive questions
        4. Be clear, specific, and directly answerable from the document
        5. Be formulated as complete questions with question marks

        Return only the questions, one per line, without any additional text, numbering, or explanations.
        """

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)

        questions_text = response.text.strip()
        questions = [q.strip() for q in questions_text.split("\n") if q.strip().endswith("?")]

        if len(questions) < count and len(questions) > 0:
            existing = "\n".join(questions)
            additional_prompt = f"""
            ### Document Content:
            {combined_text}

            ### Existing Questions:
            {existing}

            ### Instructions:
            Based on the document content provided above, generate {count - len(questions)} more important and insightful questions that are different from the existing questions listed.

            Return only the new questions, one per line, without any additional text or numbering.
            """

            additional_response = model.generate_content(additional_prompt)
            additional_questions = [q.strip() for q in additional_response.text.strip().split("\n") if q.strip().endswith("?")]
            questions.extend(additional_questions)

        return questions[:count]

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return [f"Error generating questions: {str(e)}"]
