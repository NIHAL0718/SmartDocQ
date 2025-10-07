"""Main Streamlit application for SmartDocQ."""

import os
import time
import json
import random
import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_mic_recorder import mic_recorder
from streamlit_feedback import streamlit_feedback
import speech_recognition as sr
from gtts import gTTS
import tempfile
from io import BytesIO
import base64

# Import utility functions
from utils.document_utils import display_document_info, display_document_list, poll_document_status
from utils.audio_utils import text_to_speech, speech_to_text
from utils.chat_utils import display_chat_history, add_message_to_history
from utils.ui_utils import show_success, show_error, show_info, apply_custom_css
from utils.translation_utils import translate_text, text_to_text_translate, get_supported_languages, detect_language, check_backend_connection
from utils.auth_utils import login_user, register_user, logout_user, is_authenticated, get_current_user

# Set page configuration
st.set_page_config(
    page_title="SmartDocQ - AI Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply modern CSS theme
apply_custom_css()

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000/api")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_document" not in st.session_state:
    st.session_state.current_document = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "important_questions" not in st.session_state:
    st.session_state.important_questions = []

if "language" not in st.session_state:
    st.session_state.language = "english"
    
# Authentication session state
if "user" not in st.session_state:
    st.session_state.user = None
    
if "token" not in st.session_state:
    st.session_state.token = None
    
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "voice_output" not in st.session_state:
    st.session_state.voice_output = False

# Initialize voice settings if not already set
if "voice_speed" not in st.session_state:
    st.session_state.voice_speed = 1.0
if "voice_pitch" not in st.session_state:
    st.session_state.voice_pitch = 1.0


def show_login_page():
    """Show login page."""
    st.header("Login")
    
    # Display registration success message if redirected from registration
    if st.session_state.get("registration_success"):
        show_success(st.session_state.get("registration_message", "Registration successful"))
        st.info("Please login with your new account.")
        # Clear the registration success flag
        st.session_state.registration_success = False
    
    # Display password requirements reminder
    st.info("Remember: Password must be at least 8 characters long and contain at least one letter and one number.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                success, message = login_user(username, password)
                if success:
                    show_success(message)
                    st.rerun()
                else:
                    show_error(message)
    
    if st.button("Don't have an account? Register", key="register_link"):
        st.session_state.nav = "Register"
        st.rerun()


def show_register_page():
    """Show registration page."""
    st.header("Register")
    
    # Display password requirements
    st.info("Password must be at least 8 characters long and contain at least one letter and one number.")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        st.caption("Username must be 3-16 characters long and can only contain letters, numbers, underscores, and hyphens.")
        email = st.text_input("Email (optional)")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Register")
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(username, password, email if email else None)
                if success:
                    # Store success message in session state to display on login page
                    st.session_state.registration_success = True
                    st.session_state.registration_message = message
                    # Redirect to login page
                    st.session_state.nav = "Login"
                    st.rerun()
                else:
                    show_error(message)
    
    if st.button("Already have an account? Login", key="login_link"):
        st.session_state.nav = "Login"
        st.rerun()


def main():
    """Main application function."""
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
        text-shadow: 0px 0px 2px rgba(0, 0, 0, 0.5);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
    }
    .assistant-message {
        background-color: #F5F5F5;
        border-left: 5px solid #424242;
        color: #000000;
    }
    .assistant-message p {
        color: #000000;
    }
    .source-info {
        font-size: 0.8rem;
        color: #616161;
        margin-top: 0.5rem;
    }
    .feedback-container {
        margin-top: 0.5rem;
    }
    .suggested-question-container {
        margin-bottom: 0.5rem;
        width: 100%;
    }
    .suggested-question {
        background-color: #E3F2FD;
        color: #0D47A1;
        border: 1px solid #1E88E5;
        border-radius: 5px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: normal;
        text-align: left;
        display: block;
        width: 100%;
        cursor: pointer;
    }
    .suggested-question:hover {
        background-color: #BBDEFB;
    }
    /* Fix for input text color */
    .stTextInput>div>div>input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    /* Fix for question box background */
    .stTextInput>div {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">SmartDocQ</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">AI-powered Document Question Answering</h2>', unsafe_allow_html=True)

    # Determine navigation: before login only Login/Register; after login show top menu
    if "nav" not in st.session_state:
        st.session_state.nav = "Login" if not is_authenticated() else "Home"

    # Before login: only login/register (no tabs to allow programmatic redirect)
    if not is_authenticated():
        if st.session_state.nav == "Register":
            show_register_page()
        else:
            st.session_state.nav = "Login"
            show_login_page()
        # Prevent double-render of forms by exiting early before router below
        return
    else:
        # After login: top navigation menu
        menu = option_menu(
            None,
            ["Home", "Upload", "Chat", "Library", "OCR", "Translate", "Settings", "Logout"],
            icons=["house", "cloud-upload", "chat", "collection", "camera", "translate", "gear", "box-arrow-right"],
            menu_icon="menu-button-wide",
            default_index=0,
            orientation="horizontal",
        )
        if menu == "Upload":
            st.session_state.nav = "Upload Document"
        elif menu == "Library":
            st.session_state.nav = "Document Library"
        elif menu == "Translate":
            st.session_state.nav = "Translation"
        elif menu == "Logout":
            logout_user()
            st.session_state.nav = "Login"
            st.rerun()
        else:
            st.session_state.nav = menu
        selected = st.session_state.nav

    # Main content based on navigation
    if selected == "Home":
        show_home_page()
    elif selected == "Login":
        show_login_page()
    elif selected == "Register":
        show_register_page()
    elif selected == "Upload Document":
        if is_authenticated():
            show_upload_page()
        else:
            st.warning("Please login to access this feature.")
            show_login_page()
    elif selected == "Chat":
        if is_authenticated():
            show_chat_page()
        else:
            st.warning("Please login to access this feature.")
            show_login_page()
    elif selected == "Document Library":
        if is_authenticated():
            show_document_library()
        else:
            st.warning("Please login to access this feature.")
            show_login_page()
    elif selected == "OCR":
        if is_authenticated():
            show_ocr_page()
        else:
            st.warning("Please login to access this feature.")
            show_login_page()
    elif selected == "Translation":
        if is_authenticated():
            show_translation_page()
        else:
            st.warning("Please login to access this feature.")
            show_login_page()
    elif selected == "Settings":
        show_settings_page()


def show_home_page():
    """Display the home page."""
    st.subheader("Welcome to SmartDocQ")
    
    # App description
    st.markdown("""
    SmartDocQ is an AI-powered document question answering application that helps you extract insights from your documents.
    
    ### Features:
    - Upload and process PDF, Word, text files, and web pages
    - Ask questions about your documents and get accurate answers
    - Voice-based interaction for questions and answers
    - Multi-language support (English, Hindi, Telugu, Tamil)
    - Chat memory to track conversation history
    - OCR support for scanned documents and handwritten notes
    - Translation service for text and documents
    
    ### Get Started:
    1. Upload a document in the **Upload Document** section
    2. Ask questions about your document in the **Chat** section
    3. Explore your document library in the **Document Library** section
    4. Process scanned documents with **OCR**
    5. Translate text or documents in the **Translation** section
    """)
    
    # Home page content without quick access buttons
    
    # Recent documents
    if st.session_state.documents:
        st.subheader("Recent Documents")
        display_document_list(st.session_state.documents[:3], set_current=True)


def show_upload_page():
    """Display the document upload page."""
    st.subheader("Upload Document")
    
    # Single File Upload (remove web page option)
    with st.container():
        # File upload form
        with st.form(key="file_upload_form"):
            uploaded_file = st.file_uploader(
                "Upload a document",
                type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
                help="Upload a PDF, Word, text file, or image (JPG, PNG)",
            )
            title = st.text_input("Document Title (optional)")
            st.info("Document language will be automatically detected using OCR for images and content analysis for text documents.")
            # Hidden language field with default value
            language = "English"
            submit_button = st.form_submit_button("Upload")
        
        if submit_button and uploaded_file is not None:
            # Check if the file is an image
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension in ["jpg", "jpeg", "png"]:
                st.info("Image detected. OCR will be used to extract text for question generation.")
                
            # Display spinner during upload
            with st.spinner("Uploading and processing document..."):
                try:
                    # Prepare the file for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), f"application/{uploaded_file.type.split('/')[-1]}")}
                    form_data = {"title": title if title else uploaded_file.name, "language": "english"}
                    
                    # Make API call to upload document
                    upload_url = f"{API_URL}/documents/upload"
                    response = requests.post(upload_url, files=files, data=form_data)
                    
                    # Check if upload was successful
                    if response.status_code == 200:
                        upload_data = response.json()
                        doc_id = upload_data.get("id")
                        doc_title = upload_data.get("title") or (title if title else uploaded_file.name)
                        
                        # Add to session state
                        new_doc = {
                            "id": doc_id,
                            "title": doc_title,
                            "upload_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "file_type": uploaded_file.name.split(".")[-1],
                            "language": upload_data.get("language", "english"),
                            # Get document statistics from the upload response if available
                            "pages": upload_data.get("page_count", 0),
                            "chunks": upload_data.get("chunk_count", 0),
                            "word_count": upload_data.get("word_count", 0),
                        }
                        
                        st.session_state.documents.append(new_doc)
                        st.session_state.current_document = new_doc
                        
                        # Wait a moment for processing
                        time.sleep(2)
                        
                        # Start polling for document status updates
                        with st.spinner("Processing document..."):
                            # Poll for status updates a few times
                            for _ in range(5):  # Try 5 times with 2-second intervals
                                if poll_document_status(doc_id):
                                    break  # Document is processed
                                time.sleep(2)  # Wait before polling again
                        
                        # Get questions from the backend
                        try:
                            questions_url = f"{API_URL}/documents/{doc_id}/questions"
                            questions_response = requests.get(questions_url)
                            
                            if questions_response.status_code == 200:
                                questions_data = questions_response.json()
                                st.session_state.important_questions = questions_data.get("questions", [])
                            else:
                                # Fallback to default questions if API call fails
                                st.session_state.important_questions = [
                                    "What are the main themes discussed in the document?",
                                    "How does the document address the key challenges?",
                                    "What solutions are proposed in the document?",
                                    "Who are the main stakeholders mentioned?",
                                    "What are the potential implications of the findings?",
                                ]
                        except Exception as e:
                            st.error(f"Error getting questions: {str(e)}")
                            # Fallback to default questions
                            st.session_state.important_questions = [
                                "What are the main themes discussed in the document?",
                                "How does the document address the key challenges?",
                                "What solutions are proposed in the document?",
                                "Who are the main stakeholders mentioned?",
                                "What are the potential implications of the findings?",
                            ]
                        
                        show_success(f"Document '{doc_title}' uploaded successfully!")
                    else:
                        show_error(f"Failed to upload document: {response.text}")
                except Exception as e:
                    show_error(f"Error uploading document: {str(e)}")
    
    
    # Display current document info if available
    if st.session_state.current_document:
        st.subheader("Current Document")
        
        # Add refresh button for document statistics
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write("")
        with col2:
            if st.button("🔄 Refresh", key="refresh_stats"):
                if poll_document_status(st.session_state.current_document["id"]):
                    st.success("Document statistics updated!")
                else:
                    st.info("Document is still processing or could not be updated.")
        
        display_document_info(st.session_state.current_document)
        
        # Display important questions
        if st.session_state.important_questions:
            st.subheader("Important Questions")
            for i, question in enumerate(st.session_state.important_questions):
                if st.button(question, key=f"q_{i}_{hash(question)}"):
                    # Set the question in the chat page and navigate there
                    st.session_state.current_question = question
                    st.session_state.page = "chat"
                    st.rerun()


def show_chat_page():
    """Display the chat interface."""
    st.subheader("Chat with your Document")
    
    # Check if a document is selected
    if not st.session_state.current_document:
        st.warning("Please upload or select a document first.")
        if st.button("Go to Upload"):
            st.session_state.page = "upload"
            st.rerun()
        return
    
    # Display current document info
    st.write(f"**Current Document:** {st.session_state.current_document['title']}")
    
    # Chat history
    display_chat_history(st.session_state.chat_history)
    
    # Check if there's a current question from the document page
    if st.session_state.current_question:
        question = st.session_state.current_question
        st.write(f"Processing question: {question}")
        st.session_state.current_question = None  # Clear it after use
        
        # Add user message to chat history
        add_message_to_history("user", question)
        
        # Display spinner during processing
        with st.spinner("Generating answer..."):
            try:
                # Prepare the data for API call
                data = {
                    "question": question,
                    "document_id": st.session_state.current_document["id"]
                }
                
                # Make API call to get the answer
                answer_url = f"{API_URL}/chat/question"
                response = requests.post(answer_url, json=data)
                
                # Check if request was successful
                if response.status_code == 200:
                    response_data = response.json()
                    answer = response_data.get("answer", "Sorry, I couldn't generate an answer.")
                    sources = response_data.get("sources", [])
                    
                    # If no sources were returned, create an empty list
                    if not sources:
                        sources = []
                    
                    # Add assistant message to chat history
                    add_message_to_history("assistant", answer, sources)
                else:
                    # Fallback to a generic answer if API call fails
                    answer = f"Sorry, I couldn't process your question. Error: {response.text}"
                    add_message_to_history("assistant", answer, [])
                    show_error(f"Failed to get answer: {response.text}")
            except Exception as e:
                # Fallback to a generic answer if an exception occurs
                answer = f"Sorry, I couldn't process your question due to an error."
                add_message_to_history("assistant", answer, [])
                show_error(f"Error getting answer: {str(e)}")
        
        # Rerun to update the UI
        st.rerun()
    
    # Question input
    st.subheader("Ask a Question")
    
    # Tabs for text and voice input
    tab1, tab2 = st.tabs(["Text", "Voice"])
    
    with tab1:
        # Text input
        with st.form(key="question_form"):
            question = st.text_input("Your question", key="text_question")
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_button = st.form_submit_button("Ask")
            with col2:
                voice_output = st.checkbox("Voice answer", value=st.session_state.get("voice_output", False), key="voice_output_checkbox")
        
        # Save voice output preference to session state
        if voice_output != st.session_state.get("voice_output", False):
            st.session_state.voice_output = voice_output
            st.rerun()
            
        if submit_button and question:
            # Add user message to chat history
            add_message_to_history("user", question)
            
            # Display spinner during processing
            with st.spinner("Generating answer..."):
                try:
                    # Prepare the data for API call
                    data = {
                        "question": question,
                        "document_id": st.session_state.current_document["id"]
                    }
                    
                    # Make API call to get the answer
                    answer_url = f"{API_URL}/chat/question"
                    response = requests.post(answer_url, json=data)
                    
                    # Check if request was successful
                    if response.status_code == 200:
                        response_data = response.json()
                        answer = response_data.get("answer", "Sorry, I couldn't generate an answer.")
                        sources = response_data.get("sources", [])
                        
                        # If no sources were returned, create an empty list
                        if not sources:
                            sources = []
                        
                        # Add assistant message to chat history
                        add_message_to_history("assistant", answer, sources)
                    else:
                        # Fallback to a generic answer if API call fails
                        answer = f"Sorry, I couldn't process your question. Error: {response.text}"
                        add_message_to_history("assistant", answer, [])
                        show_error(f"Failed to get answer: {response.text}")
                except Exception as e:
                    # Fallback to a generic answer if an exception occurs
                    answer = f"Sorry, I couldn't process your question due to an error."
                    add_message_to_history("assistant", answer, [])
                    show_error(f"Error getting answer: {str(e)}")
                
                # Generate voice answer if requested
                if st.session_state.voice_output:
                    audio_data = text_to_speech(answer, st.session_state.language)
                    st.audio(audio_data, format="audio/mp3")
            
            # Rerun to update the UI
            st.rerun()
    
    with tab2:
        # Voice input
        st.write("Click the microphone button and speak your question")
        
        # Record audio
        audio_bytes = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="voice_recorder")
        
        if audio_bytes:
            # Display spinner during processing
            with st.spinner("Processing voice input..."):
                try:
                    # Convert audio to text using the speech_to_text function
                    question = speech_to_text(audio_bytes)
                    
                    if not question:
                        st.error("Could not understand audio. Please try again.")
                        return
                    
                    st.write(f"**Your question:** {question}")
                    
                    # Add user message to chat history
                    add_message_to_history("user", question)
                    
                    # Prepare the data for API call
                    data = {
                        "question": question,
                        "document_id": st.session_state.current_document["id"]
                    }
                    
                    # Make API call to get the answer
                    answer_url = f"{API_URL}/chat/question"
                    response = requests.post(answer_url, json=data)
                    
                    # Check if request was successful
                    if response.status_code == 200:
                        response_data = response.json()
                        answer = response_data.get("answer", "Sorry, I couldn't generate an answer.")
                        sources = response_data.get("sources", [])
                        
                        # If no sources were returned, create an empty list
                        if not sources:
                            sources = []
                        
                        # Add assistant message to chat history
                        add_message_to_history("assistant", answer, sources)
                    else:
                        # Fallback to a generic answer if API call fails
                        answer = f"Sorry, I couldn't process your question. Error: {response.text}"
                        add_message_to_history("assistant", answer, [])
                        show_error(f"Failed to get answer: {response.text}")
                except Exception as e:
                    # Fallback to a generic answer if an exception occurs
                    answer = f"Sorry, I couldn't process your question due to an error."
                    add_message_to_history("assistant", answer, [])
                    show_error(f"Error getting answer: {str(e)}")
                
                # Voice answer is already added to chat history above
                
                # Generate voice answer
                audio_data = text_to_speech(answer, st.session_state.language)
                st.audio(audio_data, format="audio/mp3")
            
            # Rerun to update the UI
            st.rerun()
    
    # Important questions suggestions
    if st.session_state.important_questions:
        st.subheader("Suggested Questions")
        cols = st.columns(2)
        for i, question in enumerate(st.session_state.important_questions):
            with cols[i % 2]:
                # Use Streamlit's button with custom styling
                if st.button(question, key=f"sugg_q_{i}", help="Click to ask this question", use_container_width=True):
                    # Add user message to chat history
                    add_message_to_history("user", question)
                    
                    # Display spinner during processing
                    with st.spinner("Generating answer..."):
                        try:
                            # Prepare the data for API call
                            data = {
                                "question": question,
                                "document_id": st.session_state.current_document["id"]
                            }
                            
                            # Make API call to get the answer
                            answer_url = f"{API_URL}/chat/question"
                            response = requests.post(answer_url, json=data)
                            
                            # Check if request was successful
                            if response.status_code == 200:
                                response_data = response.json()
                                answer = response_data.get("answer", "Sorry, I couldn't generate an answer.")
                                sources = response_data.get("sources", [])
                                
                                # If no sources were returned, create an empty list
                                if not sources:
                                    sources = []
                                
                                # Add assistant message to chat history
                                add_message_to_history("assistant", answer, sources)
                                
                                # Generate voice answer if voice output is enabled
                                if st.session_state.get("voice_output", False):
                                    audio_data = text_to_speech(answer, st.session_state.language)
                                    st.audio(audio_data, format="audio/mp3")
                            else:
                                # Fallback to a generic answer if API call fails
                                answer = f"Sorry, I couldn't process your question. Error: {response.text}"
                                add_message_to_history("assistant", answer, [])
                                show_error(f"Failed to get answer: {response.text}")
                        except Exception as e:
                            # Fallback to a generic answer if an exception occurs
                            answer = f"Sorry, I couldn't process your question due to an error."
                            add_message_to_history("assistant", answer, [])
                            show_error(f"Error getting answer: {str(e)}")
                    
                    # Rerun to update the UI
                    st.rerun()
    
    # Chat export options
    if st.session_state.chat_history:
        st.subheader("Export Chat")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export as PDF"):
                from utils.chat_utils import export_chat_history
                pdf_bytes = export_chat_history("pdf")
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="smartdocq_chat.pdf",
                    mime="application/pdf",
                    key="download_chat_pdf",
                )
        with col2:
            if st.button("Export as Word"):
                from utils.chat_utils import export_chat_history
                docx_bytes = export_chat_history("docx")
                st.download_button(
                    label="Download Word",
                    data=docx_bytes,
                    file_name="smartdocq_chat.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="download_chat_docx",
                )


def show_document_library():
    """Display the document library."""
    st.subheader("Document Library")
    
    # Search and filter
    search_query = st.text_input("Search documents", placeholder="Enter keywords...")
    
    # Filter documents if search query is provided
    documents = st.session_state.documents
    if search_query:
        documents = [doc for doc in documents if search_query.lower() in doc["title"].lower()]
    
    # Display documents
    if documents:
        display_document_list(documents, set_current=True, show_actions=True)
    else:
        st.info("No documents found. Upload a document to get started.")


def process_ocr(file, enhance_image=True):
    """Process an image with OCR using the backend API.
    
    Args:
        file: The uploaded file object
        enhance_image: Whether to enhance the image for better OCR results
        
    Returns:
        dict: OCR result with text, confidence, and language
    """
    from utils.translation_utils import API_URL, check_backend_connection
    import requests
    
    # Check if backend is accessible
    if not check_backend_connection():
        st.error("⚠️ Backend server is not accessible. Please check if the server is running.")
        return {"status": "error", "message": "Backend connection failed"}
    
    try:
        # Prepare the file for upload
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"enhance_image": str(enhance_image).lower()}
        
        # Call the OCR API
        response = requests.post(
            f"{API_URL}/ocr/process",
            files=files,
            data=data,
            timeout=300  # Increased timeout to 5 minutes for large images or slow processing
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Error processing OCR: {response.text}")
            return {"status": "error", "message": f"API error: {response.status_code}"}
    
    except Exception as e:
        st.error(f"Error processing OCR: {str(e)}")
        return {"status": "error", "message": str(e)}


def show_ocr_page():
    """Display the OCR page."""
    st.subheader("Optical Character Recognition (OCR)")
    
    # OCR upload form
    with st.form(key="ocr_form"):
        uploaded_file = st.file_uploader(
            "Upload an image or scanned document",
            type=["jpg", "jpeg", "png", "pdf"],
            help="Upload an image or scanned PDF",
        )
        
        # OCR options (removed visible enhance checkbox; default behavior remains True)
        enhance_image = True
        
        # Note that language selection is removed as it will be auto-detected
        st.info("Language will be automatically detected from the document content.")
        
        submit_button = st.form_submit_button("Process with OCR")
    
    if submit_button and uploaded_file is not None:
        # Display spinner during processing
        with st.spinner("Processing with OCR..."):
            # Call the OCR API
            ocr_result = process_ocr(uploaded_file, enhance_image)
            
            if ocr_result.get("status") == "error":
                st.error(f"OCR processing failed: {ocr_result.get('message', 'Unknown error')}")
                return
            
            # Extract OCR results
            ocr_text = ocr_result.get("text", "")
            confidence = ocr_result.get("confidence", 0.0)
            language = ocr_result.get("language", "unknown")
            
            # Display the result
            st.subheader("OCR Result")
            st.text_area("Extracted Text", ocr_text, height=200)
            
            # Confidence score
            st.write(f"**Confidence Score:** {confidence:.1%}")
            
            # Detected language
            st.write(f"**Detected Language:** {language.capitalize()}")
            
            # Options for the extracted text
            st.subheader("What would you like to do with this text?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create Searchable Document"):
                    # Create a document from the OCR text
                    with st.spinner("Creating document..."):
                        doc_id = f"ocr-{int(time.time())}"
                        doc_title = uploaded_file.name.split(".")[0]
                        
                        # Add to session state
                        new_doc = {
                            "id": doc_id,
                            "title": f"OCR: {doc_title}",
                            "upload_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "file_type": "ocr",
                            "language": language,
                            "text": ocr_text,
                        }
                        
                        if "documents" not in st.session_state:
                            st.session_state.documents = []
                        
                        st.session_state.documents.append(new_doc)
                        st.session_state.current_document = new_doc
                        
                        show_success("Document created from OCR text!")
            with col2:
                if st.button("Copy to Clipboard"):
                    # Use JavaScript to copy text to clipboard
                    st.write("<script>navigator.clipboard.writeText('" + ocr_text.replace("'", "\\'").replace("\n", "\\n") + "');</script>", unsafe_allow_html=True)
                    show_info("Text copied to clipboard.")


def show_settings_page():
    """Display the settings page."""
    st.subheader("Settings")
    
    # Language settings
    st.write("### Language Settings")
    interface_language = st.selectbox(
        "Interface Language",
        ["English", "Hindi", "Telugu", "Tamil"],
        index=["english", "hindi", "telugu", "tamil"].index(st.session_state.language),
    )
    st.session_state.language = interface_language.lower()
    
    # Voice settings
    st.write("### Voice Settings")
    voice_speed = st.slider("Voice Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    voice_pitch = st.slider("Voice Pitch", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    
    # Save settings
    if st.button("Save Settings"):
        # In a real app, this would save the settings
        show_success("Settings saved successfully!")
    
    # Clear data
    st.write("### Data Management")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        show_success("Chat history cleared!")
    
    if st.button("Clear Document Library"):
        st.session_state.documents = []
        st.session_state.current_document = None
        st.session_state.important_questions = []
        show_success("Document library cleared!")


def show_translation_page():
    """Display the translation page."""
    st.subheader("Translation Service")
    
    # Initialize session state for translation
    if "translation_history" not in st.session_state:
        st.session_state.translation_history = []
    
    # Initialize text-to-text translation preference
    if "use_text_to_text" not in st.session_state:
        st.session_state.use_text_to_text = True
    
    # Removed backend connectivity banner for cleaner UI
    
    # Get supported languages
    languages = get_supported_languages()
    
    # Create a dictionary mapping language names to codes for easy lookup
    language_options = {lang["name"]: lang["code"] for lang in languages}
    
    # Create tabs for different translation modes
    tab1, tab2, tab3 = st.tabs(["Text Translation", "Current Document Translation", "Upload Document Translation"])
    
    with tab1:
        # Text translation form
        with st.form(key="text_translation_form"):
            # Source text input
            source_text = st.text_area(
                "Enter text to translate",
                height=150,
                placeholder="Type or paste text here..."
            )
            
            # Language selection
            col1, col2 = st.columns(2)
            with col1:
                source_lang = st.selectbox(
                    "Source Language",
                    ["Auto Detect"] + [lang["name"] for lang in languages],
                    index=0,
                )
            with col2:
                target_lang = st.selectbox(
                    "Target Language",
                    [lang["name"] for lang in languages],
                    index=0,
                )
            
            # Translation method selection
            use_text_to_text = st.checkbox("Use Text-to-Text Translation", value=st.session_state.use_text_to_text, 
                                         help="Enable this for improved translation quality using direct text-to-text translation",
                                         key="use_text_to_text")
            
            # Submit button
            submit_button = st.form_submit_button("Translate")
        
        if submit_button and source_text:
            # Get language codes
            target_lang_code = language_options[target_lang]
            source_lang_code = None if source_lang == "Auto Detect" else language_options[source_lang]
            
            # Display spinner during translation
            with st.spinner("Translating..."):
                # Call appropriate translation API based on selected method
                if use_text_to_text:
                    result = text_to_text_translate(source_text, target_lang_code, source_lang_code)
                else:
                    result = translate_text(source_text, target_lang_code, source_lang_code)
                
                # Check if there was an error in the translation
                if "error" in result:
                    error_message = result["error"]
                    # If the input and output are the same and there's an error, it might be a translation issue
                    if source_text == result["translated_text"] and not "fallback translation" in error_message.lower():
                        st.error("⚠️ Translation failed: The translated text is identical to the input.")
                        st.info("Try using simpler phrases or common words for better results.")
                else:
                    # Display result
                    st.subheader("Translation Result")
                    st.text_area("Translated Text", result["translated_text"], height=150)
                    
                    # Display detected source language if auto-detect was used
                    if source_lang == "Auto Detect" and "source_language" in result:
                        detected_lang = next((lang["name"] for lang in languages if lang["code"] == result["source_language"]), result["source_language"])
                        st.info(f"Detected language: {detected_lang}")
                
                # Add to translation history only if there's no error or if we want to keep track of errors too
                st.session_state.translation_history.append({
                    "source_text": source_text,
                    "translated_text": result["translated_text"],
                    "source_language": result.get("source_language", source_lang_code),
                    "target_language": target_lang_code,
                    "translation_method": "Text-to-Text" if use_text_to_text else "Standard",
                    "timestamp": time.time(),
                    "error": result.get("error", None)
                })
        
        # Text-to-speech for translation result
        if st.session_state.translation_history:
            latest_translation = st.session_state.translation_history[-1]
            if st.button("Listen to Translation", key="listen_text_translation"):
                with st.spinner("Generating audio..."):
                    audio_data = text_to_speech(latest_translation["translated_text"], latest_translation["target_language"])
                    st.audio(audio_data, format="audio/mp3")
    
    with tab2:
        # Current document translation
        if st.session_state.current_document:
            st.write(f"**Current Document:** {st.session_state.current_document['title']}")
            
            # Get document content from backend
            try:
                import os
                import requests
                
                # Get API URL from environment or use default
                API_URL = os.getenv("API_URL", "http://localhost:8000/api")
                
                # Make API call to get document content
                doc_id = st.session_state.current_document['id']
                content_url = f"{API_URL}/documents/{doc_id}/content"
                
                with st.spinner("Loading document content..."):
                    content_response = requests.get(content_url)
                    
                    if content_response.status_code == 200:
                        content_data = content_response.json()
                        document_text = content_data.get("content", "")
                        
                        # Display document content preview
                        st.subheader("Document Content Preview")
                        st.text_area("Original Content", document_text[:1000] + ("..." if len(document_text) > 1000 else ""), height=150)
                        
                        # Language selection for current document
                        col1, col2 = st.columns(2)
                        with col1:
                            curr_doc_source_lang = st.selectbox(
                                "Source Language",
                                ["Auto Detect"] + [lang["name"] for lang in languages],
                                index=0,
                                key="curr_doc_source_lang"
                            )
                        with col2:
                            curr_doc_target_lang = st.selectbox(
                                "Target Language",
                                [lang["name"] for lang in languages],
                                index=0,
                                key="curr_doc_target_lang"
                            )
                        
                        # Translation method selection
                        st.checkbox("Use Text-to-Text Translation", value=st.session_state.use_text_to_text, 
                                  help="Enable this for improved translation quality using direct text-to-text translation",
                                  key="use_text_to_text_doc")
                        
                        # Translate button
                        if st.button("Translate Document", key="translate_current_doc"):
                            # Get language codes
                            target_lang_code = language_options[curr_doc_target_lang]
                            source_lang_code = None if curr_doc_source_lang == "Auto Detect" else language_options[curr_doc_source_lang]
                            
                            # Display spinner during translation
                            with st.spinner("Translating document..."):
                                # Call appropriate translation API based on selected method
                                if st.session_state.get('use_text_to_text', True):
                                    result = text_to_text_translate(document_text, target_lang_code, source_lang_code)
                                else:
                                    result = translate_text(document_text, target_lang_code, source_lang_code)
                                
                                # Display result
                                st.subheader("Translation Result")
                                translated_text = result["translated_text"]
                                st.text_area("Translated Content", translated_text[:1000] + ("..." if len(translated_text) > 1000 else ""), height=150)
                                
                                # Display detected source language if auto-detect was used
                                if curr_doc_source_lang == "Auto Detect" and "source_language" in result:
                                    detected_lang = next((lang["name"] for lang in languages if lang["code"] == result["source_language"]), result["source_language"])
                                    st.info(f"Detected language: {detected_lang}")
                                
                                # Add to translation history
                                st.session_state.translation_history.append({
                                    "source_text": f"Document: {st.session_state.current_document['title']}",
                                    "translated_text": translated_text,
                                    "source_language": result.get("source_language", source_lang_code),
                                    "target_language": target_lang_code,
                                    "timestamp": time.time()
                                })
                                
                                # Offer download option
                                st.download_button(
                                    label="Download Translated Document",
                                    data=translated_text,
                                    file_name=f"translated_{st.session_state.current_document['title']}.txt",
                                    mime="text/plain"
                                )
                    else:
                        st.error(f"Error loading document content: {content_response.status_code}")
                        st.info("Please make sure the backend server is running and the document is properly processed.")
            except Exception as e:
                st.error(f"Error accessing document content: {str(e)}")
                st.info("Please make sure the backend server is running and the document is properly processed.")
        else:
            st.info("No document is currently selected. Please select a document from the Document Library or upload a new document.")
            if st.button("Go to Document Library"):
                st.session_state.page = "document_library"
                st.rerun()
    
    with tab3:
        # Document translation form for uploaded files
        with st.form(key="document_translation_form"):
            uploaded_file = st.file_uploader(
                "Upload a document to translate",
                type=["pdf", "docx", "txt"],
                help="Upload a PDF, Word, or text file",
            )
            
            # Language selection
            col1, col2 = st.columns(2)
            with col1:
                doc_source_lang = st.selectbox(
                    "Source Language",
                    ["Auto Detect"] + [lang["name"] for lang in languages],
                    index=0,
                    key="doc_source_lang"
                )
            with col2:
                doc_target_lang = st.selectbox(
                    "Target Language",
                    [lang["name"] for lang in languages],
                    index=0,
                    key="doc_target_lang"
                )
            
            # Translation method selection
            st.checkbox("Use Text-to-Text Translation", value=st.session_state.use_text_to_text, 
                      help="Enable this for improved translation quality using direct text-to-text translation",
                      key="use_text_to_text_upload")
            
            # Submit button
            submit_doc_button = st.form_submit_button("Translate Document")
        
        if submit_doc_button and uploaded_file is not None:
            # Display spinner during processing
            with st.spinner("Processing document for translation..."):
                try:
                    # Read the uploaded file
                    file_content = uploaded_file.read()
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    # Extract text based on file type
                    document_text = ""
                    if file_extension == ".pdf":
                        import PyPDF2
                        from io import BytesIO
                        pdf_file = BytesIO(file_content)
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        for page in pdf_reader.pages:
                            document_text += page.extract_text() + "\n"
                    elif file_extension == ".docx":
                        import docx
                        from io import BytesIO
                        doc_file = BytesIO(file_content)
                        doc = docx.Document(doc_file)
                        for para in doc.paragraphs:
                            document_text += para.text + "\n"
                    elif file_extension == ".txt":
                        document_text = file_content.decode("utf-8")
                    
                    # Get language codes
                    target_lang_code = language_options[doc_target_lang]
                    source_lang_code = None if doc_source_lang == "Auto Detect" else language_options[doc_source_lang]
                    
                    # Call appropriate translation API based on selected method
                    if st.session_state.get('use_text_to_text', True):
                        result = text_to_text_translate(document_text, target_lang_code, source_lang_code)
                    else:
                        result = translate_text(document_text, target_lang_code, source_lang_code)
                    
                    # Check if there was an error in the translation
                    if "error" in result:
                        error_message = result["error"]
                        # If the input and output are the same and there's an error, it's likely an API key issue
                        if document_text == result["translated_text"]:
                            st.error("⚠️ Translation failed: The translated text is identical to the input.")
                            if "API key" in error_message:
                                st.warning("The translation service requires a valid Google Gemini API key. Please contact the administrator to update the API key.")
                                st.info("Administrator: You can obtain a free API key from https://ai.google.dev/")
                    else:
                        # Display result
                        st.subheader("Translation Result")
                        translated_text = result["translated_text"]
                        st.text_area("Translated Content", translated_text[:1000] + ("..." if len(translated_text) > 1000 else ""), height=150)
                        
                        # Display detected source language if auto-detect was used
                        if doc_source_lang == "Auto Detect" and "source_language" in result:
                            detected_lang = next((lang["name"] for lang in languages if lang["code"] == result["source_language"]), result["source_language"])
                            st.info(f"Detected language: {detected_lang}")
                        
                        # Offer download option
                        st.download_button(
                            label="Download Translated Document",
                            data=translated_text,
                            file_name=f"translated_{uploaded_file.name.split('.')[0]}.txt",
                            mime="text/plain"
                        )
                    
                    # Add to translation history regardless of error
                    st.session_state.translation_history.append({
                        "source_text": f"Uploaded Document: {uploaded_file.name}",
                        "translated_text": result["translated_text"],
                        "source_language": result.get("source_language", source_lang_code),
                        "target_language": target_lang_code,
                        "timestamp": time.time(),
                        "error": result.get("error", None)
                    })
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    st.info("Please make sure the file is valid and try again.")
    
    # Translation history
    if st.session_state.translation_history:
        with st.expander("Translation History"):
            for i, item in enumerate(reversed(st.session_state.translation_history)):
                st.write(f"**Translation {i+1}**")
                st.write(f"**Source:** {item['source_text'][:100]}{'...' if len(item['source_text']) > 100 else ''}")
                st.write(f"**Translation:** {item['translated_text'][:100]}{'...' if len(item['translated_text']) > 100 else ''}")
                
                # Get language names
                try:
                    source_lang_name = next((lang["name"] for lang in languages if lang["code"] == item["source_language"]), item["source_language"])
                    target_lang_name = next((lang["name"] for lang in languages if lang["code"] == item["target_language"]), item["target_language"])
                    st.write(f"**Languages:** {source_lang_name} → {target_lang_name}")
                    
                    # Display translation method if available
                    if "translation_method" in item:
                        st.write(f"**Method:** {item['translation_method']}")
                except (TypeError, KeyError):
                    # Fallback if there's an issue with language lookup
                    st.write(f"**Languages:** {item.get('source_language', 'unknown')} → {item.get('target_language', 'unknown')}")
                st.write("---")
        
        # Clear history button
        if st.button("Clear Translation History"):
            st.session_state.translation_history = []
            st.experimental_rerun()


if __name__ == "__main__":
    main()