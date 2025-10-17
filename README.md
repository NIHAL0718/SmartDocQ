# SmartDocQ - AI-powered Document Question Answering Application

SmartDocQ is an advanced document question answering application that leverages AI to provide intelligent answers based on document content. The application supports various document formats, multiple languages, and offers voice-based interaction.

## Features

- **Document Upload**: Support for PDF, Word, text files, and web pages
- **Semantic Search**: Documents are chunked and embedded for semantic search
- **Vector Database**: Fast retrieval of relevant document sections
- **AI-Powered Answers**: Uses Google Gemini for generating accurate answers
- **Voice Interaction**: Ask questions by voice and receive spoken answers
- **Multi-language Support**: English, Hindi, Telugu, and Tamil
- **Chat Memory**: Conversation history with export options (PDF/Word)
- **Feedback System**: Rate answers or report errors
- **Important Question Generation**: Automatically generates key questions from documents
- **OCR Support**: Reads scanned documents and handwritten notes

## Project Structure

```
SmartDocQ/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core functionality
│   │   ├── models/         # Data models
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utility functions
│   ├── main.py             # FastAPI application entry point
│   └── requirements.txt    # Backend dependencies
├── frontend/               # Streamlit frontend
│   ├── pages/              # Application pages
│   ├── components/         # UI components
│   ├── utils/              # Frontend utilities
│   ├── app.py              # Streamlit application entry point
│   └── requirements.txt    # Frontend dependencies
└── README.md               # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- Virtual environment (recommended)

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

## API Documentation

Once the backend server is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technologies Used

- **Backend**: FastAPI, Google Gemini API, PyPDF2, python-docx, BeautifulSoup, FAISS/Chroma DB
- **Frontend**: Streamlit, SpeechRecognition, gTTS
- **Data Processing**: LangChain, Sentence Transformers
- **OCR**: Tesseract, EasyOCR

## License

MIT