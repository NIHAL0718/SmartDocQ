"""Utility functions for document handling in the frontend."""

import streamlit as st
import time
import os
import requests


def poll_document_status(doc_id):
    """Poll for document status updates.
    
    Args:
        doc_id (str): Document ID to poll for status
        
    Returns:
        bool: True if document is processed, False otherwise
    """
    try:
        # Get API URL from environment or use default
        API_URL = os.getenv("API_URL", "http://localhost:8000/api")
        
        # Make API call to get document status
        status_url = f"{API_URL}/documents/status/{doc_id}"
        response = requests.get(status_url)
        
        if response.status_code == 200:
            status_data = response.json()
            
            # Update document statistics in session state
            if st.session_state.current_document and st.session_state.current_document["id"] == doc_id:
                st.session_state.current_document["pages"] = status_data.get("page_count", 0)
                st.session_state.current_document["chunks"] = status_data.get("chunk_count", 0)
                st.session_state.current_document["word_count"] = status_data.get("word_count", 0)
                
                # Also update in the documents list
                for doc in st.session_state.documents:
                    if doc["id"] == doc_id:
                        doc["pages"] = status_data.get("page_count", 0)
                        doc["chunks"] = status_data.get("chunk_count", 0)
                        doc["word_count"] = status_data.get("word_count", 0)
                        break
            
            # Return True if document is processed
            return status_data.get("status") == "completed"
        
        return False
    except Exception as e:
        print(f"Error polling document status: {e}")
        return False


def display_document_info(document):
    """Display detailed information about a document.
    
    Args:
        document (dict): Document information dictionary
    """
    # Create columns for document info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Title:** {document['title']}")
        st.write(f"**ID:** {document['id']}")
        st.write(f"**Type:** {document['file_type'].upper()}")
    
    with col2:
        st.write(f"**Upload Date:** {format_date(document.get('upload_date', ''))}")
        if 'language' in document:
            st.write(f"**Language:** {document['language'].capitalize()}")
        if 'source_url' in document:
            st.write(f"**Source URL:** {document['source_url']}")
    
    # Document statistics
    st.write("### Document Statistics")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Pages", str(document.get('pages', 'N/A')))
    with stats_col2:
        st.metric("Chunks", str(document.get('chunks', 'N/A')))
    with stats_col3:
        st.metric("Word Count", str(document.get('word_count', 'N/A')))


def display_document_list(documents, set_current=False, show_actions=False):
    """Display a list of documents with optional actions.
    
    Args:
        documents (list): List of document dictionaries
        set_current (bool): Whether to allow setting current document
        show_actions (bool): Whether to show delete and other actions
    """
    for i, doc in enumerate(documents):
        # Create a container for each document
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{doc['title']}**")
                st.write(f"Type: {doc['file_type'].upper()} | Date: {format_date(doc.get('upload_date', ''))}")
            
            with col2:
                if set_current:
                    if st.button("Select", key=f"select_{doc['id']}"):
                        st.session_state.current_document = doc
                        
                        # Poll for document status updates
                        poll_document_status(doc['id'])
                        
                        # Get questions from the backend API
                        try:
                            import os
                            import requests
                            
                            # Get API URL from environment or use default
                            API_URL = os.getenv("API_URL", "http://localhost:8000/api")
                            
                            # Make API call to get questions
                            questions_url = f"{API_URL}/documents/{doc['id']}/questions"
                            with st.spinner("Loading questions..."):
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
                        
                        st.rerun()
            
            with col3:
                if show_actions:
                    if st.button("Delete", key=f"delete_{doc['id']}"):
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
            
            # Add a separator
            st.markdown("---")


def format_date(date_str):
    """Format date string for display.
    
    Args:
        date_str (str): ISO format date string
        
    Returns:
        str: Formatted date string
    """
    if not date_str:
        return "Unknown"
    
    try:
        # Parse ISO format date
        date_obj = time.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        return time.strftime("%b %d, %Y %H:%M", date_obj)
    except ValueError:
        return date_str