"""Main Streamlit application for SmartDocQ."""

import os
import re
import time
import json
import random
import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_feedback import streamlit_feedback
import tempfile
from io import BytesIO
import base64

# Import utility functions
from utils.document_utils import display_document_info, display_document_list, poll_document_status
from utils.audio_utils import text_to_speech, speech_to_text
from utils.chat_utils import display_chat_history, add_message_to_history
from utils.ui_utils import show_success, show_error, show_info, apply_custom_css, create_modern_header, create_glass_card, create_document_card, create_chat_message_bubble, create_loading_spinner, create_stats_grid
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

# Initialize voice settings if not already set (kept for compatibility but unused)
if "voice_speed" not in st.session_state:
    st.session_state.voice_speed = 1.0
if "voice_pitch" not in st.session_state:
    st.session_state.voice_pitch = 1.0
if "show_upload_spinner" not in st.session_state:
    st.session_state.show_upload_spinner = False


def load_documents_from_backend():
    """Fetch documents from backend and store in session state."""
    try:
        resp = requests.get(f"{API_URL}/documents/list", timeout=15)
        if resp.status_code == 200:
            items = resp.json() or []
            normalized = []
            for it in items:
                normalized.append({
                    "id": it.get("id"),
                    "title": it.get("title"),
                    "upload_date": it.get("upload_date", time.strftime("%Y-%m-%dT%H:%M:%S")),
                    "file_type": it.get("file_type", "unknown"),
                    "language": it.get("language", "english"),
                    "pages": it.get("page_count", it.get("pages", 0)),
                    "chunks": it.get("chunk_count", it.get("chunks", 0)),
                    "word_count": it.get("word_count", 0),
                })
            st.session_state.documents = normalized
    except Exception:
        # Silent fail; keep existing session documents
        pass

def show_login_page():
    """Show modern login page."""
    # Simplified header removed per request
    
    # Display registration success message if redirected from registration
    if st.session_state.get("registration_success"):
        st.markdown('<div class="glass-card" style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid var(--success); margin-bottom: 2rem;">', unsafe_allow_html=True)
        show_success(st.session_state.get("registration_message", "Registration successful"))
        st.info("Please login with your new account.")
        st.markdown('</div>', unsafe_allow_html=True)
        # Clear the registration success flag
        st.session_state.registration_success = False
    
    # Login form with glassmorphism styling
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    
    with st.form("login_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                success, message = login_user(username, password)
                if success:
                    show_success(message)
                    load_documents_from_backend()
                    st.rerun()
                else:
                    show_error(message)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Register link centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Register", key="register_link", use_container_width=True):
            st.session_state.nav = "Register"
            st.rerun()


def show_register_page():
    """Show modern registration page."""
    # Simplified header removed per request
    
    # Registration form with glassmorphism styling
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    
    with st.form("register_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email (optional)", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", placeholder="Create a secure password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            submit_button = st.form_submit_button("Create Account", use_container_width=True)
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                # Validate email format if provided
                if email:
                    email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
                    if not re.match(email_pattern, email):
                        st.error("Invalid email format")
                        st.stop()
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Login link centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Login", key="login_link", use_container_width=True):
            st.session_state.nav = "Login"
            st.rerun()


def main():
    """Main application function."""
    # Modern header with gradient text
    st.markdown(create_modern_header("SmartDocQ", "AI-powered Document Question Answering"), unsafe_allow_html=True)

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
        items = ["Home", "Upload", "Chat", "Library", "OCR", "Translate", "Logout"]
        idx_map = {"Home": 0, "Upload Document": 1, "Chat": 2, "Document Library": 3, "OCR": 4, "Translation": 5, "Logout": 6}
        # Keep default index synced with current route
        current_index = idx_map.get(st.session_state.get("nav", "Home"), 0)
        previous_nav = st.session_state.get("nav", "Home")
        menu = option_menu(
            None,
            items,
            icons=["house", "cloud-upload", "chat", "collection", "camera", "translate", "box-arrow-right"],
            menu_icon="menu-button-wide",
            default_index=current_index,
            orientation="horizontal",
        )
        # Respect programmatic nav override for a single rerun
        if st.session_state.get("nav_override"):
            selected = st.session_state.nav
            st.session_state.nav_override = False
        else:
            # Map menu label to internal route consistently
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
        # If navigation changed due to user click, rerun immediately so content updates without second click
        if selected != previous_nav:
            st.rerun()

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
    # Settings page removed per request


def show_home_page():
    """Display the modern home page dashboard."""
    # Welcome section
    st.markdown('<div class="glass-card fade-in" style="text-align: center; margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: var(--text-primary); margin-bottom: 1rem;">🏠 Welcome to SmartDocQ</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); font-size: 1.1rem; line-height: 1.6;">Transform your documents into intelligent conversations with our AI-powered question answering system.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features grid
    features = [
        {"icon": "📄", "title": "Document Upload", "description": "Upload PDFs, Word docs, text files, and images"},
        {"icon": "💬", "title": "AI Chat", "description": "Ask questions and get intelligent answers from your documents"},
        {"icon": "🔍", "title": "OCR Processing", "description": "Extract text from scanned documents and handwritten notes"},
        {"icon": "🌐", "title": "Translation", "description": "Translate text and documents across multiple languages"},
        {"icon": "📚", "title": "Document Library", "description": "Organize and manage all your uploaded documents"},
        {"icon": "🎯", "title": "Smart Questions", "description": "Get AI-generated questions to explore your documents"}
    ]
    
    st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">✨ Features</h3>', unsafe_allow_html=True)
    
    # Create feature cards in a grid
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            st.markdown(f'''
            <div class="glass-card fade-in" style="text-align: center; padding: 1.5rem; margin-bottom: 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">{feature["icon"]}</div>
                <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">{feature["title"]}</h4>
                <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.4;">{feature["description"]}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Quick stats
    if st.session_state.documents:
        total_docs = len(st.session_state.documents)
        total_pages = sum(doc.get('pages', 0) for doc in st.session_state.documents)
        total_words = sum(doc.get('word_count', 0) for doc in st.session_state.documents)
        
        stats = [
            {"icon": "📊", "label": "Total Documents", "value": str(total_docs)},
            {"icon": "📄", "label": "Total Pages", "value": str(total_pages)},
            {"icon": "📝", "label": "Total Words", "value": f"{total_words:,}"},
            {"icon": "💬", "label": "Chat Messages", "value": str(len(st.session_state.chat_history))}
        ]
        
        st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">📈 Your Stats</h3>', unsafe_allow_html=True)
        st.components.v1.html(create_stats_grid(stats), height=260)
    
    # Recent documents with modern cards
    if st.session_state.documents:
        st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">📚 Recent Documents</h3>', unsafe_allow_html=True)
        
        recent_docs = st.session_state.documents[:3]
        cols = st.columns(len(recent_docs))
        
        for i, doc in enumerate(recent_docs):
            with cols[i]:
                st.components.v1.html(create_document_card(doc), height=220)
                
                # Add select button
                if st.button(f"Select {doc['title'][:20]}...", key=f"select_recent_{doc['id']}", use_container_width=True):
                    st.session_state.current_document = doc
                    st.rerun()
    
    # Quick actions removed per request


def show_upload_page():
    """Display the modern document upload page."""
    # Modern header
    st.markdown('<h2 style="color: var(--text-primary); margin-bottom: 2rem;">📤 Upload Document</h2>', unsafe_allow_html=True)
    
    # Upload form with glassmorphism styling
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    
    with st.form(key="file_upload_form"):
        st.markdown('<h3 style="text-align: center; color: var(--text-primary); margin-bottom: 2rem;">📁 Choose Your Document</h3>', unsafe_allow_html=True)
        
        # File uploader with modern styling
        uploaded_file = st.file_uploader(
            "📄 Upload a document",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
            help="Supported formats: PDF, Word, Text files, and Images (JPG, PNG)",
        )
        
        # Document details (language option removed)
        col1, col2 = st.columns([1,1])
        with col1:
            title = st.text_input("📝 Document Title (optional)", placeholder="Enter a custom title")
        with col2:
            st.markdown(" ")
            st.markdown(" ")
        
        # Upload button
        submit_button = st.form_submit_button("🚀 Upload & Process", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_button and uploaded_file is not None:
        # Check if the file is an image
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ["jpg", "jpeg", "png"]:
            st.info("🖼️ Image detected. OCR will be used to extract text for question generation.")
        
        # Modern progress indicator
        st.session_state.show_upload_spinner = True
        progress_container = st.container()
        with progress_container:
            st.markdown(create_loading_spinner("Uploading and processing document..."), unsafe_allow_html=True)

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
                    "pages": upload_data.get("page_count", 0),
                    "chunks": upload_data.get("chunk_count", 0),
                    "word_count": upload_data.get("word_count", 0),
                }
                
                st.session_state.documents.append(new_doc)
                st.session_state.current_document = new_doc
                
                # Start polling for document status updates with spinner, then clear upload spinner
                with st.spinner("Processing document..."):
                    for _ in range(8):  # Try 8 times with 1.5-second intervals
                        if poll_document_status(doc_id):
                            break
                        time.sleep(1.5)
                # Clear upload progress container
                st.session_state.show_upload_spinner = False
                progress_container.empty()
                
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
                
                show_success(f"✅ Document '{doc_title}' uploaded successfully!")
                # Refresh backend list and recent docs
                load_documents_from_backend()
                # Rerun to ensure any lingering spinners are removed
                st.rerun()
            else:
                show_error(f"❌ Failed to upload document: {response.text}")
        except Exception as e:
            show_error(f"❌ Error uploading document: {str(e)}")
        finally:
            # Always clear spinner if still visible
            try:
                st.session_state.show_upload_spinner = False
                progress_container.empty()
            except Exception:
                pass
    
    # Display current document info if available
    if st.session_state.current_document:
        st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">📄 Current Document</h3>', unsafe_allow_html=True)
        
        # Modern document card
        st.components.v1.html(create_document_card(st.session_state.current_document), height=220)
        
        # Refresh button
        col1, col2, col3 = st.columns([4, 1, 4])
        with col2:
            if st.button("🔄 Refresh Stats", key="refresh_stats", use_container_width=True):
                if poll_document_status(st.session_state.current_document["id"]):
                    st.success("Document statistics updated!")
                else:
                    st.info("Document is still processing or could not be updated.")
        
        # Display important questions with modern styling
        if st.session_state.important_questions:
            st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">🎯 Suggested Questions</h3>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
            for i, question in enumerate(st.session_state.important_questions):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f'<p style="color: var(--text-secondary); margin: 0.5rem 0; padding: 0.75rem; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 3px solid var(--accent-primary);">{question}</p>', unsafe_allow_html=True)
                with col2:
                    if st.button("Ask", key=f"q_{i}_{hash(question)}", use_container_width=True):
                        # Redirect to Chat with the selected question
                        st.session_state.current_question = question
                        st.session_state.nav = "Chat"
                        st.session_state.nav_override = True
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def show_chat_page():
    """Display the modern chat interface."""
    # Modern header
    st.markdown('<h2 style="color: var(--text-primary); margin-bottom: 2rem;">💬 Chat with your Document</h2>', unsafe_allow_html=True)
    
    # Check if a document is selected
    if not st.session_state.current_document:
        st.markdown('<div class="glass-card fade-in" style="text-align: center; padding: 2rem;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--text-primary); margin-bottom: 1rem;">No Document Selected</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: var(--text-secondary); margin-bottom: 2rem;">Please upload or select a document first to start chatting.</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📤 Upload Document", key="go_to_upload", use_container_width=True):
                st.session_state.nav = "Upload Document"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Current document info with modern styling
    st.markdown('<div class="glass-card fade-in" style="margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 1rem;"><span style="font-size: 1.5rem; margin-right: 0.5rem;">📄</span><h3 style="color: var(--text-primary); margin: 0;">{st.session_state.current_document["title"]}</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: var(--text-secondary); margin: 0;">Type: {st.session_state.current_document.get("file_type", "Unknown").upper()} • Pages: {st.session_state.current_document.get("pages", "N/A")} • Words: {st.session_state.current_document.get("word_count", "N/A")}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat history with modern styling
    if st.session_state.chat_history:
        st.markdown('<h3 style="color: var(--text-primary); margin-bottom: 1rem;">💭 Conversation History</h3>', unsafe_allow_html=True)
        
        # Chat container with modern styling
        st.markdown('<div class="chat-container fade-in">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            st.markdown(create_chat_message_bubble(
                message["role"], 
                message["content"], 
                message.get("timestamp"),
                message.get("sources")
            ), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card fade-in" style="text-align: center; padding: 2rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 3rem; margin-bottom: 1rem;">💭</div>', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--text-primary); margin-bottom: 1rem;">Start the Conversation</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: var(--text-secondary);">Ask questions about your document below to get intelligent answers.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if there's a current question from the document page
    if st.session_state.current_question:
        question = st.session_state.current_question
        st.markdown(f'<div class="glass-card" style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid var(--accent-secondary); margin-bottom: 1rem;"><p style="color: var(--text-primary); margin: 0;"><strong>Processing question:</strong> {question}</p></div>', unsafe_allow_html=True)
        st.session_state.current_question = None  # Clear it after use
        
        # Add user message to chat history
        add_message_to_history("user", question)
        
        # Display spinner during processing
        with st.spinner("🤖 Generating answer..."):
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
    
    # Modern question input form
    st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">❓ Ask a Question</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    with st.form(key="question_form"):
        question = st.text_input("💭 Your question", key="text_question", placeholder="Ask anything about your document...")
        submit_button = st.form_submit_button("🚀 Ask Question", use_container_width=True)

    if submit_button and question:
        # Add user message to chat history
        add_message_to_history("user", question)

        # Display spinner during processing
        with st.spinner("🤖 Generating answer..."):
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat export with modern styling
    if st.session_state.chat_history:
        st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">📥 Export Chat</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        from utils.chat_utils import export_chat_history
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Prepare PDF", key="prepare_pdf", use_container_width=True):
                st.session_state["chat_pdf_bytes"] = export_chat_history("pdf")
                st.success("PDF ready for download!")
        with col2:
            if st.button("📝 Prepare DOCX", key="prepare_docx", use_container_width=True):
                st.session_state["chat_docx_bytes"] = export_chat_history("docx")
                st.success("DOCX ready for download!")
        
        # Download buttons
        if st.session_state.get("chat_pdf_bytes"):
            st.download_button(
                label="📄 Download Chat PDF", 
                data=st.session_state["chat_pdf_bytes"], 
                file_name="smartdocq_chat.pdf", 
                mime="application/pdf",
                use_container_width=True
            )
        if st.session_state.get("chat_docx_bytes"):
            st.download_button(
                label="📝 Download Chat DOCX", 
                data=st.session_state["chat_docx_bytes"], 
                file_name="smartdocq_chat.docx", 
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)


def show_document_library():
    """Display the modern document library."""
    # Modern header
    st.markdown('<h2 style="color: var(--text-primary); margin-bottom: 2rem;">📚 Document Library</h2>', unsafe_allow_html=True)
    
    # Search section removed per request
    
    # No filtering; show all documents
    documents = st.session_state.documents.copy()
    
    # Default order preserved (no sort option)
    
    # Display documents with modern cards
    if documents:
        st.markdown(f'<h3 style="color: var(--text-primary); margin-bottom: 1rem;">📄 Found {len(documents)} document(s)</h3>', unsafe_allow_html=True)
        
        # Create a grid of document cards
        cols_per_row = 3
        for i in range(0, len(documents), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, doc in enumerate(documents[i:i+cols_per_row]):
                with cols[j]:
                    st.components.v1.html(create_document_card(doc), height=220)
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📄 Select", key=f"select_{doc['id']}", use_container_width=True):
                            st.session_state.current_document = doc
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{doc['id']}", use_container_width=True):
                            # Remove document from session state
                            st.session_state.documents.remove(doc)
                            # If current document is deleted, clear it
                            if st.session_state.current_document and st.session_state.current_document['id'] == doc['id']:
                                st.session_state.current_document = None
                                st.session_state.important_questions = []
                                # Also clear chat history related to this document
                                try:
                                    if 'chat_history' in st.session_state:
                                        st.session_state.chat_history = []
                                except Exception:
                                    pass
                            st.rerun()
    else:
        st.markdown('<div class="glass-card fade-in" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--text-primary); margin-bottom: 1rem;">No Documents Found</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: var(--text-secondary); margin-bottom: 2rem;">Upload your first document to get started with SmartDocQ.</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📤 Upload Document", key="upload_from_library", use_container_width=True):
                st.session_state.nav = "Upload Document"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


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
        # Downscale large images client-side to speed up OCR
        try:
            from PIL import Image
            from io import BytesIO
            if hasattr(file, 'getvalue'):
                raw_bytes = file.getvalue()
            else:
                raw_bytes = file.read()
            img = Image.open(BytesIO(raw_bytes))
            max_dim = 1600
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim))
                buf = BytesIO()
                img.save(buf, format=img.format or 'PNG', optimize=True, quality=85)
                buf.seek(0)
                files = {"file": (file.name, buf.getvalue(), file.type)}
            else:
                files = {"file": (file.name, file.getvalue(), file.type)}
        except Exception:
            # Fallback to original upload if PIL not available or on error
            files = {"file": (file.name, file.getvalue(), file.type)}

        # Prepare the file for upload
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
    """Display the modern OCR page."""
    # Modern header
    st.markdown('<h2 style="color: var(--text-primary); margin-bottom: 2rem;">🔍 Optical Character Recognition</h2>', unsafe_allow_html=True)
    
    # OCR description
    st.markdown('<div class="glass-card fade-in" style="margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: var(--text-primary); margin-bottom: 1rem;">📸 Extract Text from Images</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); line-height: 1.6;">Upload scanned documents, handwritten notes, or images to extract text using advanced OCR technology. The system will automatically detect the language and provide accurate text extraction.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # OCR upload form with modern styling
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    
    with st.form(key="ocr_form"):
        st.markdown('<h3 style="text-align: center; color: var(--text-primary); margin-bottom: 2rem;">📁 Choose Your Image</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "🖼️ Upload an image or scanned document",
            type=["jpg", "jpeg", "png", "pdf"],
            help="Supported formats: JPG, PNG, PDF. Language will be auto-detected.",
        )
        
        # OCR options
        col1, col2 = st.columns(2)
        with col1:
            enhance_image = st.checkbox("🔧 Enhance Image", value=False, help="Apply image enhancement for better OCR results (slower)")
        with col2:
            auto_detect = st.checkbox("🌐 Auto-detect Language", value=True, help="Automatically detect the text language")
        
        submit_button = st.form_submit_button("🚀 Process with OCR", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_button and uploaded_file is not None:
        # Display image preview
        st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">🖼️ Image Preview</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        
        # Modern progress indicator
        st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">⚙️ Processing</h3>', unsafe_allow_html=True)
        st.markdown(create_loading_spinner("Processing with OCR..."), unsafe_allow_html=True)
        
        # Call the OCR API
        ocr_result = process_ocr(uploaded_file, enhance_image)
        
        if ocr_result.get("status") == "error":
            st.error(f"❌ OCR processing failed: {ocr_result.get('message', 'Unknown error')}")
            return
        
        # Extract OCR results
        ocr_text = ocr_result.get("text", "")
        language = ocr_result.get("language", "unknown")

        # Display only extracted text (remove metrics and buttons)
        st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">📝 Extracted Text</h3>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        if ocr_text:
            st.text_area("", ocr_text, height=300, key="ocr_result_text")
        else:
            st.markdown('<div style="text-align: center; padding: 2rem; color: var(--text-secondary);">No text could be extracted from this image.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


 


def show_translation_page():
    """Display the modern translation page."""
    # Modern header
    st.markdown('<h2 style="color: var(--text-primary); margin-bottom: 2rem;">🌐 Translation Service</h2>', unsafe_allow_html=True)
    
    # Initialize session state for translation
    if "translation_history" not in st.session_state:
        st.session_state.translation_history = []
    
    # Initialize text-to-text translation preference
    if "use_text_to_text" not in st.session_state:
        st.session_state.use_text_to_text = True
    
    # Intro description removed to eliminate extra container under header
    
    # Get supported languages
    languages = get_supported_languages()
    
    # Create a dictionary mapping language names to codes for easy lookup
    language_options = {lang["name"]: lang["code"] for lang in languages}
    
    # Create modern tabs for different translation modes
    tab1, tab2, tab3 = st.tabs(["📝 Text Translation", "📄 Current Document", "📁 Upload Document"])
    
    with tab1:
        # Text translation form with modern styling
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        
        with st.form(key="text_translation_form"):
            st.markdown('<h3 style="text-align: center; color: var(--text-primary); margin-bottom: 2rem;">📝 Translate Text</h3>', unsafe_allow_html=True)
            
            # Source text input
            source_text = st.text_area(
                "📄 Enter text to translate",
                height=150,
                placeholder="Type or paste text here..."
            )
            
            # Language selection
            col1, col2 = st.columns(2)
            with col1:
                source_lang = st.selectbox(
                    "🌐 Source Language",
                    ["Auto Detect"] + [lang["name"] for lang in languages],
                    index=0,
                )
            with col2:
                target_lang = st.selectbox(
                    "🎯 Target Language",
                    [lang["name"] for lang in languages],
                    index=0,
                )
            
            # Translation method selection
            use_text_to_text = st.checkbox("🔧 Use Enhanced Translation", 
                                         help="Enable this for improved translation quality using advanced AI translation",
                                         key="use_text_to_text")
            
            # Submit button
            submit_button = st.form_submit_button("🚀 Translate", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submit_button and source_text:
            # Get language codes
            target_lang_code = language_options[target_lang]
            source_lang_code = None if source_lang == "Auto Detect" else language_options[source_lang]
            
            # Scoped spinner while translating
            with st.spinner("Translating text..."):
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
                # Display result with modern styling
                st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">✅ Translation Result</h3>', unsafe_allow_html=True)
                
                st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
                st.text_area("📄 Translated Text", result["translated_text"], height=150, key="translation_result")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔊 Listen", key="listen_translation", use_container_width=True):
                        with st.spinner("Generating audio..."):
                            audio_data = text_to_speech(result["translated_text"], target_lang_code)
                            st.audio(audio_data, format="audio/mp3")
                
                with col2:
                    st.download_button(
                        label="📥 Download",
                        data=result["translated_text"],
                        file_name=f"translation_{target_lang_code}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Add/overwrite a persistent latest translation entry
            st.session_state.translation_history.append({
                "source_text": source_text,
                "translated_text": result["translated_text"],
                "source_language": result.get("source_language", source_lang_code),
                "target_language": target_lang_code,
                "translation_method": "Enhanced" if use_text_to_text else "Standard",
                "timestamp": time.time(),
                "error": result.get("error", None)
            })
        
        # Translation history with modern styling
        if st.session_state.translation_history:
            st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">📚 Translation History</h3>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
            # Show latest translation
            latest_translation = st.session_state.translation_history[-1]
            st.markdown(f'<h4 style="color: var(--text-primary); margin-bottom: 1rem;">Latest Translation</h4>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<p style="color: var(--text-secondary);"><strong>Source:</strong> {latest_translation["source_text"][:100]}{"..." if len(latest_translation["source_text"]) > 100 else ""}</p>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<p style="color: var(--text-secondary);"><strong>Translation:</strong> {latest_translation["translated_text"][:100]}{"..." if len(latest_translation["translated_text"]) > 100 else ""}</p>', unsafe_allow_html=True)
            
            # Action buttons for latest translation
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔊 Listen", key="listen_latest_translation", use_container_width=True):
                    with st.spinner("Generating audio..."):
                        audio_data = text_to_speech(latest_translation["translated_text"], latest_translation["target_language"])
                        st.audio(audio_data, format="audio/mp3")
            with col2:
                st.download_button(
                    label="📥 Download",
                    data=latest_translation["translated_text"],
                    file_name=f"latest_translation.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col3:
                if st.button("🗑️ Clear History", key="clear_translation_history", use_container_width=True):
                    st.session_state.translation_history = []
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Current document translation with modern styling
        if st.session_state.current_document:
            st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
            st.markdown(f'<h3 style="color: var(--text-primary); margin-bottom: 1rem;">📄 Current Document: {st.session_state.current_document["title"]}</h3>', unsafe_allow_html=True)
            
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
                        st.markdown('<h4 style="color: var(--text-primary); margin: 1.5rem 0 1rem 0;">📄 Document Content Preview</h4>', unsafe_allow_html=True)
                        st.text_area("Original Content", document_text[:1000] + ("..." if len(document_text) > 1000 else ""), height=150, key="doc_preview")
                        
                        # Language selection for current document
                        col1, col2 = st.columns(2)
                        with col1:
                            curr_doc_source_lang = st.selectbox(
                                "🌐 Source Language",
                                ["Auto Detect"] + [lang["name"] for lang in languages],
                                index=0,
                                key="curr_doc_source_lang"
                            )
                        with col2:
                            curr_doc_target_lang = st.selectbox(
                                "🎯 Target Language",
                                [lang["name"] for lang in languages],
                                index=0,
                                key="curr_doc_target_lang"
                            )
                        
                        # Translation method selection
                        use_enhanced = st.checkbox("🔧 Use Enhanced Translation", 
                                  help="Enable this for improved translation quality using advanced AI translation",
                                  key="use_text_to_text_doc")
                        
                        # Translate button
                        if st.button("🚀 Translate Document", key="translate_current_doc", use_container_width=True):
                            # Get language codes
                            target_lang_code = language_options[curr_doc_target_lang]
                            source_lang_code = None if curr_doc_source_lang == "Auto Detect" else language_options[curr_doc_source_lang]
                            
                            # Scoped spinner while translating document
                            with st.spinner("Translating document..."):
                                if use_enhanced:
                                    result = text_to_text_translate(document_text, target_lang_code, source_lang_code)
                                else:
                                    result = translate_text(document_text, target_lang_code, source_lang_code)
                            
                            # Display result with modern styling
                            st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">✅ Translation Result</h3>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
                            translated_text = result["translated_text"]
                            st.text_area("📄 Translated Content", translated_text[:1000] + ("..." if len(translated_text) > 1000 else ""), height=150, key="doc_translation_result")
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📥 Download Translation",
                                    data=translated_text,
                                    file_name=f"translated_{st.session_state.current_document['title']}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            with col2:
                                if st.button("🔊 Listen", key="listen_doc_translation", use_container_width=True):
                                    with st.spinner("Generating audio..."):
                                        audio_data = text_to_speech(translated_text[:500], target_lang_code)  # Limit for audio
                                        st.audio(audio_data, format="audio/mp3")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add to translation history
                            st.session_state.translation_history.append({
                                "source_text": f"Document: {st.session_state.current_document['title']}",
                                "translated_text": translated_text,
                                "source_language": result.get("source_language", source_lang_code),
                                "target_language": target_lang_code,
                                "translation_method": "Enhanced" if use_enhanced else "Standard",
                                "timestamp": time.time()
                            })
                    else:
                        st.error("❌ Could not load document content. Please try again.")
            except Exception as e:
                st.error(f"❌ Error accessing document content: {str(e)}")
                st.info("Please make sure the backend server is running and the document is properly processed.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass-card fade-in" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 4rem; margin-bottom: 1rem;">📄</div>', unsafe_allow_html=True)
            st.markdown('<h3 style="color: var(--text-primary); margin-bottom: 1rem;">No Document Selected</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color: var(--text-secondary); margin-bottom: 2rem;">Please select a document from the Document Library or upload a new document to translate.</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📚 Go to Document Library", key="go_to_library_from_translation", use_container_width=True):
                    st.session_state.nav = "Document Library"
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Document translation form for uploaded files with modern styling
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        
        with st.form(key="document_translation_form"):
            st.markdown('<h3 style="text-align: center; color: var(--text-primary); margin-bottom: 2rem;">📁 Upload Document to Translate</h3>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "📄 Upload a document to translate",
                type=["pdf", "docx", "txt"],
                help="Supported formats: PDF, Word, or text file",
            )
            
            # Language selection
            col1, col2 = st.columns(2)
            with col1:
                doc_source_lang = st.selectbox(
                    "🌐 Source Language",
                    ["Auto Detect"] + [lang["name"] for lang in languages],
                    index=0,
                    key="doc_source_lang"
                )
            with col2:
                doc_target_lang = st.selectbox(
                    "🎯 Target Language",
                    [lang["name"] for lang in languages],
                    index=0,
                    key="doc_target_lang"
                )
            
            # Translation method selection
            use_enhanced_upload = st.checkbox("🔧 Use Enhanced Translation", 
                      help="Enable this for improved translation quality using advanced AI translation",
                      key="use_text_to_text_upload")
            
            # Submit button
            submit_doc_button = st.form_submit_button("🚀 Translate Document", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submit_doc_button and uploaded_file is not None:
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
                
                # Scoped spinner while translating uploaded document
                with st.spinner("Translating document..."):
                    if use_enhanced_upload:
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
                    # Display result with modern styling
                    st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">✅ Translation Result</h3>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
                    translated_text = result["translated_text"]
                    st.text_area("📄 Translated Content", translated_text[:1000] + ("..." if len(translated_text) > 1000 else ""), height=150, key="upload_translation_result")
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Translation",
                            data=translated_text,
                            file_name=f"translated_{uploaded_file.name.split('.')[0]}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("🔊 Listen", key="listen_upload_translation", use_container_width=True):
                            with st.spinner("Generating audio..."):
                                audio_data = text_to_speech(translated_text[:500], target_lang_code)  # Limit for audio
                                st.audio(audio_data, format="audio/mp3")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Add to translation history regardless of error
                st.session_state.translation_history.append({
                    "source_text": f"Uploaded Document: {uploaded_file.name}",
                    "translated_text": result["translated_text"],
                    "source_language": result.get("source_language", source_lang_code),
                    "target_language": target_lang_code,
                    "translation_method": "Enhanced" if use_enhanced_upload else "Standard",
                    "timestamp": time.time(),
                    "error": result.get("error", None)
                })
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.info("Please make sure the file is valid and try again.")
    
    # Translation history with modern styling (only show if there are multiple translations)
    if len(st.session_state.translation_history) > 1:
        st.markdown('<h3 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">📚 Full Translation History</h3>', unsafe_allow_html=True)
        
        with st.expander("View All Translations", expanded=False):
            st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
            # Enumerate in reverse but keep indices to allow deletion
            for idx in range(len(st.session_state.translation_history) - 1, -1, -1):
                item = st.session_state.translation_history[idx]
                
                st.markdown(f'<h4 style="color: var(--text-primary); margin-bottom: 1rem;">Translation {len(st.session_state.translation_history) - idx}</h4>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<p style="color: var(--text-secondary);"><strong>Source:</strong> {item["source_text"][:100]}{"..." if len(item["source_text"]) > 100 else ""}</p>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<p style="color: var(--text-secondary);"><strong>Translation:</strong> {item["translated_text"][:100]}{"..." if len(item["translated_text"]) > 100 else ""}</p>', unsafe_allow_html=True)
                
                try:
                    source_lang_name = next((lang["name"] for lang in languages if lang["code"] == item["source_language"]), item["source_language"])
                    target_lang_name = next((lang["name"] for lang in languages if lang["code"] == item["target_language"]), item["target_language"])
                    st.markdown(f'<p style="color: var(--text-muted);"><strong>Languages:</strong> {source_lang_name} → {target_lang_name}</p>', unsafe_allow_html=True)
                    if "translation_method" in item:
                        st.markdown(f'<p style="color: var(--text-muted);"><strong>Method:</strong> {item["translation_method"]}</p>', unsafe_allow_html=True)
                except (TypeError, KeyError):
                    st.markdown(f'<p style="color: var(--text-muted);"><strong>Languages:</strong> {item.get("source_language", "unknown")} → {item.get("target_language", "unknown")}</p>', unsafe_allow_html=True)
                
                col_del, _ = st.columns([1, 5])
                with col_del:
                    if st.button("🗑️ Delete", key=f"delete_translation_{idx}", use_container_width=True):
                        del st.session_state.translation_history[idx]
                        st.rerun()
                
                if idx > 0:  # Don't add separator after the last item
                    st.markdown('<hr style="border: 1px solid var(--border); margin: 1rem 0;">', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()