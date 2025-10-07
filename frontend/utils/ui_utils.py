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
    """Apply modern, professional dark/light CSS to the Streamlit app."""
    st.markdown("""
    <style>
      /* Hide Deploy button */
      button[kind="header"] {
        display: none !important;
      }
      
      /* Hide all Streamlit toolbar buttons */
      .stToolbar {
        display: none !important;
      }
      
      /* Hide the Deploy button specifically */
      button[data-testid="stToolbarActions"] {
        display: none !important;
      }
      
      :root {
        --bg-main: #0f172a;          /* slate-900 */
        --bg-elev: #111827;         /* gray-900 */
        --bg-card: #0b1220;         /* deep card */
        --text-primary: #e5e7eb;    /* gray-200 */
        --text-secondary: #9ca3af;  /* gray-400 */
        --accent: #22d3ee;          /* cyan-400 */
        --accent-strong: #06b6d4;   /* cyan-500 */
        --muted: #1f2937;           /* gray-800 */
        --border: #1f2937;          /* gray-800 */
        --success: #10b981;         /* emerald-500 */
        --warning: #f59e0b;         /* amber-500 */
        --error: #ef4444;           /* red-500 */
      }

      /* App background */
      .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,0.06), transparent 60%),
                    radial-gradient(900px 500px at 90% 10%, rgba(34,197,94,0.06), transparent 60%),
                    var(--bg-main) !important;
        color: var(--text-primary);
      }

      /* Sidebar */
      /* Hidden by default per product design */
      section[data-testid="stSidebar"] { display: none !important; }

      /* Cards and containers */
      div.block-container { padding-top: 1.2rem; }
      .css-1r6slb0, .css-12oz5g7, .stMarkdown, .stText, .stTextArea, .stSelectbox, .stMultiSelect { color: var(--text-primary) !important; }

      /* Buttons */
      .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        color: #0b1220; /* dark text for contrast */
        border: 0;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(34,211,238,0.25);
        transition: transform .08s ease, box-shadow .2s ease, filter .15s ease;
      }
      .stButton>button:hover, .stDownloadButton>button:hover { transform: translateY(-1px); filter: brightness(0.98); }
      .stButton>button:active, .stDownloadButton>button:active { transform: translateY(0); }

      /* Inputs */
      .stTextInput>div>div>input,
      .stTextArea textarea,
      .stSelectbox>div>div>div,
      .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
      }

      /* Tabs */
      button[role="tab"] { color: var(--text-secondary) !important; }
      button[role="tab"][aria-selected="true"] {
        color: var(--text-primary) !important;
        background: linear-gradient(180deg, rgba(34,211,238,0.15), transparent);
        border-bottom: 2px solid var(--accent);
      }

      /* Headers */
      h1, h2, h3, h4 { color: var(--text-primary) !important; }
      .main-header { font-size: 2.2rem; font-weight: 700; margin: .25rem 0 1rem 0; }
      .sub-header { font-size: 1.2rem; color: var(--text-secondary); margin-bottom: .75rem; }

      /* Chat bubbles */
      .chat-message { padding: 1rem; border-radius: 12px; margin-bottom: .6rem; border: 1px solid var(--border); }
      .user-message { background: #e6f3ff; }
      .assistant-message { background: #f3f4f6; }
      .source-info { font-size: .85rem; color: var(--text-secondary); margin-top: .4rem; }

      /* Alerts */
      .stAlert { border-radius: 12px; }

      /* Metrics and small elements */
      .stMetric { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: .75rem; }

      /* Footer space */
      .app-footer-space { height: 24px; }
    </style>
    """, unsafe_allow_html=True)


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