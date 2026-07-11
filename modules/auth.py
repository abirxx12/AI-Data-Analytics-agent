"""
Authentication Module
Handles user login and session management for the AI-Based Data Analytics Agent
"""

import streamlit as st
import hashlib
import hmac
from datetime import datetime

# Predefined users and passwords (in production, use a database)
USERS_DATABASE = {
    "admin": "admin123",
    "user": "password123",
    "analyst": "analyst@2024"
}


def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256
    
    Args:
        password (str): Plain text password
        
    Returns:
        str: Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify if plain password matches hashed password
    
    Args:
        plain_password (str): Plain text password
        hashed_password (str): Hashed password from database
        
    Returns:
        bool: True if password matches, False otherwise
    """
    return hash_password(plain_password) == hashed_password


def login_user(username: str, password: str) -> bool:
    """
    Authenticate user with username and password
    
    Args:
        username (str): User's username
        password (str): User's password
        
    Returns:
        bool: True if authentication successful, False otherwise
    """
    if username in USERS_DATABASE:
        stored_password = USERS_DATABASE[username]
        # Simple comparison for demo (in production, use hashed passwords from DB)
        if password == stored_password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.login_time = datetime.now()
            return True
    return False


def logout_user():
    """Logout the current user and clear session"""
    if "logged_in" in st.session_state:
        st.session_state.logged_in = False
    if "username" in st.session_state:
        del st.session_state.username
    if "login_time" in st.session_state:
        del st.session_state.login_time


def display_login_form() -> tuple:
    """
    Display login form and return username and password
    
    Returns:
        tuple: (username, password)
    """
    st.set_page_config(page_title="AI Analytics Agent - Login", layout="centered")
    
    # Custom styling for login page
    st.markdown("""
        <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stContainer {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🤖 AI Analytics Agent")
        st.subheader("Intelligent Data Analysis System")
        st.markdown("---")
        
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username"
        )
        password = st.text_input(
            "Password",
            placeholder="Enter your password",
            type="password",
            key="login_password"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            login_button = st.button("🔓 Login", use_container_width=True)
        
        with col_btn2:
            st.button("ℹ️ Demo Credentials", use_container_width=True, 
                     help="Username: admin, Password: admin123")
        
        st.markdown("---")
        st.info("📌 Demo Credentials:\n- **Username:** admin\n- **Password:** admin123")
        
        return username, password, login_button
    
    return "", "", False


def check_login_status():
    """Check if user is logged in, redirect to login if not"""
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        return False
    return True


def display_logout_button():
    """Display logout button in sidebar"""
    if st.session_state.get("logged_in", False):
        with st.sidebar:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"👤 **{st.session_state.username}**")
            with col2:
                if st.button("🚪 Logout", use_container_width=True):
                    logout_user()
                    st.rerun()
