"""Utility functions for UI components in the frontend."""

import streamlit as st


def show_success(message):
    """Display a success message.
    
    Args:
        message (str): Success message to display
    """
    st.success(message)


def show_error(message):
    """Display an error message.
    
    Args:
        message (str): Error message to display
    """
    st.error(message)


def show_info(message):
    """Display an info message.
    
    Args:
        message (str): Info message to display
    """
    st.info(message)


def show_warning(message):
    """Display a warning message.
    
    Args:
        message (str): Warning message to display
    """
    st.warning(message)


def create_progress_bar(title, progress=0):
    """Create a progress bar with a title.
    
    Args:
        title (str): Title for the progress bar
        progress (float): Initial progress value (0-1)
        
    Returns:
        tuple: (progress_bar, progress_text) Streamlit elements
    """
    st.write(title)
    progress_bar = st.progress(progress)
    progress_text = st.empty()
    return progress_bar, progress_text


def update_progress(progress_bar, progress_text, progress, text=""):
    """Update a progress bar and its text.
    
    Args:
        progress_bar: Streamlit progress bar element
        progress_text: Streamlit empty element for text
        progress (float): Progress value (0-1)
        text (str): Text to display
    """
    progress_bar.progress(progress)
    if text:
        progress_text.text(text)


def create_expandable_section(title, content, expanded=False):
    """Create an expandable section with a title and content.
    
    Args:
        title (str): Title for the section
        content (str): Content to display in the section
        expanded (bool): Whether the section is expanded by default
    """
    with st.expander(title, expanded=expanded):
        st.write(content)


def create_tabs(tab_names):
    """Create tabs with the given names.
    
    Args:
        tab_names (list): List of tab names
        
    Returns:
        list: List of tab objects
    """
    return st.tabs(tab_names)


def create_columns(num_columns):
    """Create a specified number of columns.
    
    Args:
        num_columns (int): Number of columns to create
        
    Returns:
        list: List of column objects
    """
    return st.columns(num_columns)


def create_sidebar_section(title):
    """Create a section in the sidebar with a title.
    
    Args:
        title (str): Title for the section
        
    Returns:
        object: Sidebar container
    """
    with st.sidebar:
        st.subheader(title)
        return st.container()


def create_file_uploader(label, file_types, key=None, help_text=None):
    """Create a file uploader with a label.
    
    Args:
        label (str): Label for the uploader
        file_types (list): List of allowed file types
        key (str, optional): Key for the uploader
        help_text (str, optional): Help text for the uploader
        
    Returns:
        object: File uploader object
    """
    return st.file_uploader(
        label,
        type=file_types,
        key=key,
        help=help_text,
    )


def create_form(key):
    """Create a form with a key.
    
    Args:
        key (str): Key for the form
        
    Returns:
        object: Form object
    """
    return st.form(key=key)


def apply_custom_css():
    """Apply modern, professional dark theme with glassmorphism effects to the Streamlit app."""
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
      
      :root {
        --bg-main: #0f172a;          /* Deep navy */
        --bg-secondary: #1e293b;     /* Slate */
        --bg-card: rgba(30, 41, 59, 0.4);  /* Glassmorphism card */
        --bg-glass: rgba(255, 255, 255, 0.05); /* Glass overlay */
        --text-primary: #f8fafc;    /* Almost white */
        --text-secondary: #cbd5e1;   /* Light gray */
        --text-muted: #94a3b8;      /* Muted gray */
        --accent-primary: #f97316;   /* Orange accent */
        --accent-secondary: #3b82f6; /* Blue accent */
        --accent-gradient: linear-gradient(135deg, #f97316, #ea580c);
        --success: #10b981;         /* Emerald */
        --warning: #f59e0b;         /* Amber */
        --error: #ef4444;           /* Red */
        --border: rgba(255, 255, 255, 0.1);
        --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        --shadow-hover: 0 12px 40px rgba(0, 0, 0, 0.4);
        --blur: blur(10px);
      }

      /* Global styles */
      * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }

      /* App background with animated gradients */
      .stApp {
        background: 
          radial-gradient(ellipse at top left, rgba(249, 115, 22, 0.1) 0%, transparent 50%),
          radial-gradient(ellipse at bottom right, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
          linear-gradient(135deg, var(--bg-main) 0%, var(--bg-secondary) 100%) !important;
        color: var(--text-primary);
        min-height: 100vh;
      }

      /* Remove any default top padding that could create empty bars */
      .main .block-container > :first-child { margin-top: 0 !important; }

      /* Hide default sidebar */
      section[data-testid="stSidebar"] { 
        display: none !important; 
      }

      /* Main container */
      .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
      }

      /* Glassmorphism cards */
      .glass-card {
        background: var(--bg-card);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
      }

      .glass-card:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
      }

      /* Modern buttons with gradients */
      .stButton > button, .stDownloadButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 20px rgba(249, 115, 22, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
        letter-spacing: 0.025em !important;
      }

      .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(249, 115, 22, 0.4) !important;
        filter: brightness(1.1) !important;
      }

      .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(0) !important;
      }

      /* Secondary buttons */
      .secondary-button {
        background: rgba(59, 130, 246, 0.1) !important;
        color: var(--accent-secondary) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        backdrop-filter: var(--blur) !important;
      }

      .secondary-button:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: var(--accent-secondary) !important;
      }

      /* Modern inputs */
      .stTextInput > div > div > input,
      .stTextArea textarea,
      .stSelectbox > div > div > div,
      .stMultiSelect > div > div > div {
        background: var(--bg-card) !important;
        color: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        backdrop-filter: var(--blur) !important;
        transition: all 0.3s ease !important;
      }

      /* Fix selectbox label and option visibility (truncate neatly) */
      .stSelectbox label { font-size: 0.9rem !important; }
      .stSelectbox [data-baseweb="select"] { min-height: 40px !important; }
      .stSelectbox [data-baseweb="select"] div { line-height: 1.2 !important; }
      .stSelectbox [data-baseweb="select"] span { 
        font-size: 0.9rem !important; 
        white-space: nowrap !important; 
        overflow: hidden !important; 
        text-overflow: ellipsis !important; 
      }
      .stSelectbox [role="listbox"] [role="option"] {
        font-size: 0.9rem !important;
        white-space: nowrap !important;
      }

      /* Hide any widget whose label is empty (removes stray blank bars) */
      label:empty { display: none !important; }
      label:empty + div { display: none !important; }
      /* Use :has to hide entire widget blocks when label is empty */
      .stTextInput:has(label:empty),
      .stTextArea:has(label:empty),
      .stSelectbox:has(label:empty),
      .stMultiSelect:has(label:empty) {
        display: none !important;
      }

      /* Ensure natural spacing after the main header */
      .main-header + .sub-header { margin-bottom: 1.25rem !important; }

      .stTextInput > div > div > input:focus,
      .stTextArea textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.1) !important;
        outline: none !important;
      }

      /* File uploader */
      .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background: var(--bg-card) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
        backdrop-filter: var(--blur) !important;
      }

      .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent-primary) !important;
        background: rgba(249, 115, 22, 0.05) !important;
      }

      /* Modern tabs - make tab list background transparent to avoid bar */
      .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-radius: 0 !important;
        padding: 0 !important;
        backdrop-filter: none !important;
        border: none !important;
        box-shadow: none !important;
      }

      .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
      }

      .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.3) !important;
      }

      /* Headers with modern typography */
      h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
      }

      .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: var(--accent-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-align: center !important;
        margin: 1rem 0 0.5rem 0 !important;
        letter-spacing: -0.025em !important;
      }

      .sub-header {
        font-size: 1.25rem !important;
        color: var(--text-secondary) !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        font-weight: 400 !important;
      }

      /* Modern chat interface */
      .chat-container {
        background: var(--bg-card) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        backdrop-filter: var(--blur) !important;
        border: 1px solid var(--border) !important;
        margin-bottom: 1rem !important;
        max-width: 960px !important;
        margin-left: auto !important;
        margin-right: auto !important;
      }

      .chat-message {
        padding: 0.9rem 1.1rem !important;
        border-radius: 14px !important;
        margin-bottom: 0.75rem !important;
        backdrop-filter: var(--blur) !important;
        border: 1px solid var(--border) !important;
        transition: all 0.3s ease !important;
        color: #ffffff !important;
        position: relative;
        overflow: hidden;
      }

      .user-message { background: rgba(59, 130, 246, 0.1) !important; border-color: rgba(59, 130, 246, 0.3) !important; margin-left: 2rem !important; }
      .user-message:after { content: ""; display: none; }

      .assistant-message { background: rgba(249, 115, 22, 0.1) !important; border-color: rgba(249, 115, 22, 0.3) !important; margin-right: 2rem !important; }
      .assistant-message:after { content: ""; display: none; }

      .chat-message:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow) !important;
      }

      /* Modern alerts */
      .stAlert {
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: var(--blur) !important;
        box-shadow: var(--shadow) !important;
      }

      .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid var(--success) !important;
      }

      .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid var(--error) !important;
      }

      .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid var(--warning) !important;
      }

      .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border-left: 4px solid var(--accent-secondary) !important;
      }

      /* Modern metrics */
      .stMetric {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        backdrop-filter: var(--blur) !important;
        box-shadow: var(--shadow) !important;
        transition: all 0.3s ease !important;
      }

      .stMetric:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-hover) !important;
      }

      /* Document cards */
      .document-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        backdrop-filter: var(--blur) !important;
        box-shadow: var(--shadow) !important;
        transition: all 0.3s ease !important;
      }

      .document-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: var(--shadow-hover) !important;
        border-color: var(--accent-primary) !important;
      }

      /* Progress bars */
      .stProgress > div > div > div > div {
        background: var(--accent-gradient) !important;
        border-radius: 8px !important;
      }

      /* Spinner */
      .stSpinner > div {
        border-color: var(--accent-primary) !important;
      }

      /* Forms */
      .stForm {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        backdrop-filter: var(--blur) !important;
        box-shadow: var(--shadow) !important;
      }

      /* Navigation menu */
      .stSelectbox > div > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        backdrop-filter: var(--blur) !important;
      }

      /* Responsive design */
      @media (max-width: 768px) {
        .main .block-container {
          padding: 1rem 0.5rem;
        }
        
        .main-header {
          font-size: 2rem !important;
        }
        
        .glass-card, .document-card {
          padding: 1rem !important;
        }
        
        .user-message, .assistant-message {
          margin-left: 0.5rem !important;
          margin-right: 0.5rem !important;
        }
      }

      /* Animations */
      @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
      }}

      @keyframes slideIn {{
        from {{ transform: translateX(-100%); }}
        to {{ transform: translateX(0); }}
      }}

      .fade-in {
        animation: fadeIn 0.6s ease-out;
      }

      .slide-in {
        animation: slideIn 0.5s ease-out;
      }

      /* Custom scrollbar */
      ::-webkit-scrollbar {
        width: 8px;
      }

      ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
      }

      ::-webkit-scrollbar-thumb {
        background: var(--accent-primary);
        border-radius: 4px;
      }

      ::-webkit-scrollbar-thumb:hover {
        background: #ea580c;
      }

      /* Loading states */
      .loading-shimmer {
        background: linear-gradient(90deg, 
          rgba(255,255,255,0.1) 25%, 
          rgba(255,255,255,0.2) 50%, 
          rgba(255,255,255,0.1) 75%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
      }

      @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
      }}
    </style>
    """, unsafe_allow_html=True)


def create_glass_card(content="", class_name="glass-card"):
    """Create a glassmorphism card container.
    
    Args:
        content (str): HTML content to display in the card
        class_name (str): CSS class name for the card
        
    Returns:
        str: HTML for the glass card
    """
    return f'<div class="{class_name} fade-in">{content}</div>'


def create_modern_header(title, subtitle="", centered=True):
    """Create a modern header with gradient text.
    
    Args:
        title (str): Main title
        subtitle (str): Subtitle text
        centered (bool): Whether to center the header
        
    Returns:
        str: HTML for the modern header
    """
    center_class = "text-align: center;" if centered else ""
    header_html = f'<div style="{center_class} margin-bottom: 2rem;">'
    header_html += f'<h1 class="main-header">{title}</h1>'
    if subtitle:
        header_html += f'<h2 class="sub-header">{subtitle}</h2>'
    header_html += '</div>'
    return header_html


def create_feature_card(icon, title, description, action_text="", action_key=""):
    """Create a modern feature card with icon and description.
    
    Args:
        icon (str): Icon or emoji for the feature
        title (str): Feature title
        description (str): Feature description
        action_text (str): Action button text
        action_key (str): Unique key for the action button
        
    Returns:
        tuple: (card_html, button_widget) if action_text provided, else card_html
    """
    card_html = f'''
    <div class="glass-card fade-in" style="text-align: center; padding: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: var(--text-primary); margin-bottom: 1rem;">{title}</h3>
        <p style="color: var(--text-secondary); line-height: 1.6;">{description}</p>
    </div>
    '''
    
    if action_text and action_key:
        button_widget = st.button(action_text, key=action_key)
        return card_html, button_widget
    
    return card_html


def create_document_card(document, show_actions=False):
    """Create a modern document card.
    
    Args:
        document (dict): Document information
        show_actions (bool): Whether to show action buttons
        
    Returns:
        str: HTML for the document card
    """
    upload_date = format_date(document.get('upload_date', ''))
    file_type = document.get('file_type', 'Unknown').upper()
    
    card_html = f'''
    <div class="document-card fade-in" style="color: #ffffff;">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem; color: #ffffff;">
            <div style="flex: 1; color: #ffffff;">
                <h4 style="color: #ffffff; margin-bottom: 0.5rem;">{document['title']}</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0;">
                    {file_type} • {upload_date}
                </p>
            </div>
            <div style="font-size: 2rem; opacity: 0.7;">{'📄' if file_type in ['PDF', 'DOCX', 'TXT'] else '🖼️'}</div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="text-align: center; border: 1px solid var(--border); border-radius: 12px; padding: 0.75rem;">
                <div style="font-size: 1.5rem; font-weight: 600; color: #ffffff;">{document.get('pages', 'N/A')}</div>
                <div style="font-size: 0.8rem; color: #ffffff;">Pages</div>
            </div>
            <div style="text-align: center; border: 1px solid var(--border); border-radius: 12px; padding: 0.75rem;">
                <div style="font-size: 1.5rem; font-weight: 600; color: #ffffff;">{document.get('chunks', 'N/A')}</div>
                <div style="font-size: 0.8rem; color: #ffffff;">Chunks</div>
            </div>
            <div style="text-align: center; border: 1px solid var(--border); border-radius: 12px; padding: 0.75rem;">
                <div style="font-size: 1.5rem; font-weight: 600; color: #ffffff;">{document.get('word_count', 'N/A')}</div>
                <div style="font-size: 0.8rem; color: #ffffff;">Words</div>
            </div>
        </div>
    </div>
    '''
    
    return card_html


def create_chat_message_bubble(role, content, timestamp=None, sources=None):
    """Create a modern chat message bubble.
    
    Args:
        role (str): Message role ('user' or 'assistant')
        content (str): Message content
        timestamp (float): Message timestamp
        sources (list): List of sources
        
    Returns:
        str: HTML for the chat message bubble
    """
    role_class = "user-message" if role == "user" else "assistant-message"
    role_icon = "👤" if role == "user" else "🤖"
    role_name = "You" if role == "user" else "AI Assistant"
    
    time_str = ""
    if timestamp:
        import time
        time_str = f'<div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">{time.strftime("%I:%M %p", time.localtime(timestamp))}</div>'
    
    # Sanitize content and sources to prevent stray HTML like </div> from breaking layout
    try:
        import html as _html
        import re as _re
    except Exception:
        _html = None
        _re = None

    safe_content = content
    if isinstance(content, str):
        text = content
        # Remove common stray HTML tags like </div> or any tag-looking fragments
        if _re is not None:
            text = _re.sub(r"</?div[^>]*>", "", text, flags=_re.IGNORECASE)
            text = _re.sub(r"</?span[^>]*>", "", text, flags=_re.IGNORECASE)
            text = _re.sub(r"</?p[^>]*>", "", text, flags=_re.IGNORECASE)
            text = _re.sub(r"<[^>]+>", "", text)  # strip any remaining tags
            # Collapse excessive blank lines/spaces
            text = _re.sub(r"\n{3,}", "\n\n", text)
        if _html is not None:
            safe_content = _html.escape(text)
        else:
            safe_content = text

    sources_html = ""
    if sources:
        sources_html = '<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);">'
        sources_html += '<div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.5rem;">📚 Sources:</div>'
        for i, source in enumerate(sources[:3]):  # Show max 3 sources
            src_text = source.get("text", "")
            if isinstance(src_text, str) and _re is not None:
                src_text = _re.sub(r"<[^>]+>", "", src_text)
            if _html is not None and isinstance(src_text, str):
                src_text = _html.escape(src_text)
            sources_html += f'<div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.25rem;">• {src_text[:100]}...</div>'
        sources_html += '</div>'
    
    message_html = f'''
    <div class="chat-message {role_class} fade-in">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{role_icon}</span>
            <span style="font-weight: 600; color: var(--text-primary);">{role_name}</span>
        </div>
        <div style="line-height: 1.6; color: var(--text-primary); white-space: pre-wrap;">{safe_content}</div>
        {time_str}
        {sources_html}
    </div>
    '''
    
    return message_html


def create_loading_spinner(text="Loading..."):
    """Create a modern loading spinner.
    
    Args:
        text (str): Loading text
        
    Returns:
        str: HTML for the loading spinner
    """
    return f'''
    <div style="text-align: center; padding: 2rem;">
        <div style="display: inline-block; width: 40px; height: 40px; border: 3px solid var(--border); border-top: 3px solid var(--accent-primary); border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 1rem;"></div>
        <p style="color: var(--text-secondary);">{text}</p>
    </div>
    <style>
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    </style>
    '''


def create_progress_indicator(current, total, label=""):
    """Create a modern progress indicator.
    
    Args:
        current (int): Current progress
        total (int): Total progress
        label (str): Progress label
        
    Returns:
        str: HTML for the progress indicator
    """
    percentage = (current / total * 100) if total > 0 else 0
    
    return f'''
    <div class="glass-card" style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: var(--text-primary); font-weight: 500;">{label}</span>
            <span style="color: var(--text-secondary);">{current}/{total}</span>
        </div>
        <div style="background: var(--bg-secondary); border-radius: 8px; height: 8px; overflow: hidden;">
            <div style="background: var(--accent-gradient); height: 100%; width: {percentage}%; transition: width 0.3s ease; border-radius: 8px;"></div>
        </div>
    </div>
    '''


def create_stats_grid(stats):
    """Create a modern stats grid.
    
    Args:
        stats (list): List of stat dictionaries with 'label', 'value', 'icon' keys
        
    Returns:
        str: HTML for the stats grid
    """
    stats_html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">'
    
    for stat in stats:
        stats_html += f'''
        <div class="glass-card" style="text-align: center; padding: 1.5rem; border: 1px solid var(--border); border-radius: 16px;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{stat.get('icon', '📊')}</div>
            <div style="font-size: 2rem; font-weight: 700; color: #ffffff; margin-bottom: 0.25rem;">{stat['value']}</div>
            <div style="font-size: 0.9rem; color: #ffffff;">{stat['label']}</div>
        </div>
        '''
    
    stats_html += '</div>'
    return stats_html


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
        import time
        # Parse ISO format date
        date_obj = time.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        return time.strftime("%b %d, %Y %H:%M", date_obj)
    except ValueError:
        return date_str


def set_page_config(title, icon, layout="wide", sidebar_state="expanded"):
    """Set the page configuration.
    
    Args:
        title (str): Page title
        icon (str): Page icon
        layout (str): Page layout
        sidebar_state (str): Initial sidebar state
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state=sidebar_state,
    )