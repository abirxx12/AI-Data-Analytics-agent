"""
AI Analytics Agent Pro - Premium SaaS Dashboard
Professional Data Analysis Platform with AI Insights
Version 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.auth import login_user, logout_user
from modules.data_processor import DataProcessor
from modules.statistical_analysis import StatisticalAnalysis
from modules.visualization import Visualizer
from modules.ml_analysis import MLAnalyzer
from modules.insight_generator import InsightGenerator
from modules.report_generator import ReportGenerator
from modules.query_analyzer import QueryAnalyzer
from modules.utils import (
    load_sample_data, get_numeric_columns, get_categorical_columns,
    create_dataframe_summary
)

# Page configuration
st.set_page_config(
    page_title="AI Analytics Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Glassmorphism CSS
st.markdown("""
    <style>
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --glass-bg: rgba(15, 23, 42, 0.95);
        --glass-bg-light: rgba(15, 23, 42, 0.8);
        --glass-border: rgba(255, 255, 255, 0.1);
        --glass-border-hover: rgba(255, 255, 255, 0.2);
        --text-primary: #ffffff;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --shadow-light: 0 4px 15px rgba(0, 0, 0, 0.1);
        --shadow-medium: 0 8px 32px rgba(0, 0, 0, 0.3);
        --shadow-heavy: 0 20px 60px rgba(0, 0, 0, 0.5);
        --border-radius: 16px;
        --border-radius-small: 8px;
    }

    * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    body {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: var(--text-primary);
        margin: 0;
        padding: 0;
        min-height: 100vh;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-gradient);
    }
    }

    body {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #ffffff;
    }

    h1, h2, h3, h4, h5, h6 {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    [data-testid="stSidebar"] {
        background: rgba(26, 26, 46, 0.95);
        border-right: 2px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }

    /* Enhanced glassmorphism cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: var(--border-radius);
        padding: 24px;
        box-shadow: var(--shadow-medium);
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--primary-gradient);
        opacity: 0.8;
    }

    .glass-card:hover {
        background: var(--glass-bg-light);
        border-color: var(--glass-border-hover);
        transform: translateY(-4px);
        box-shadow: var(--shadow-heavy);
    }

    /* Premium KPI cards */
    .kpi-card {
        background: var(--glass-bg);
        border: 2px solid transparent;
        border-radius: var(--border-radius);
        padding: 20px;
        text-align: center;
        box-shadow: var(--shadow-light);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--primary-gradient);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .kpi-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }

    .kpi-card:hover::before {
        opacity: 0.1;
    }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 12px 0 8px 0;
        line-height: 1.2;
    }

    .kpi-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        opacity: 0.9;
    }

    /* Enhanced buttons */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: var(--border-radius-small);
        font-weight: 600;
        padding: 12px 24px;
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
        font-size: 14px;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
        background: var(--secondary-gradient);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-light);
    }

    /* Secondary buttons */
    .btn-secondary {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid var(--glass-border);
        color: var(--text-primary);
    }

    .btn-secondary:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: var(--glass-border-hover);
    }

    /* Success/Warning/Error states */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-left: 4px solid #22c55e;
        border-radius: var(--border-radius-small);
        padding: 16px 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }

    .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 4px solid #ef4444;
        border-radius: var(--border-radius-small);
        padding: 16px 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }

    .stInfo {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-left: 4px solid #667eea;
        border-radius: var(--border-radius-small);
        padding: 16px 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }

    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        padding: 12px;
        border-radius: var(--border-radius);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: var(--border-radius-small);
        color: var(--text-secondary);
        font-weight: 600;
        padding: 12px 20px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.08);
        color: var(--text-primary);
        transform: translateY(-1px);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary-gradient);
        color: white;
        box-shadow: var(--shadow-light);
        border-color: rgba(102, 126, 234, 0.5);
    }

    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
        margin-right: 8px;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Enhanced sidebar */
    [data-testid="stSidebar"] {
        background: var(--glass-bg);
        border-right: 1px solid var(--glass-border);
        backdrop-filter: blur(20px);
        box-shadow: var(--shadow-medium);
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--glass-border);
        border-radius: var(--border-radius-small);
        color: var(--text-primary);
        padding: 12px 16px;
        font-size: 14px;
    }

    .stTextInput > div > div > input:focus {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }

    /* Select box styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--glass-border);
        border-radius: var(--border-radius-small);
    }

    /* Dataframe styling */
    .stDataFrame {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: var(--border-radius);
        overflow: hidden;
    }

    .stDataFrame thead th {
        background: var(--primary-gradient);
        color: white;
        font-weight: 600;
        padding: 12px;
    }

    .stDataFrame tbody td {
        padding: 12px;
        border-bottom: 1px solid var(--glass-border);
    }

    .main-container {
        max-width: 1350px;
        margin: 0 auto;
        padding: 24px 18px 48px 18px;
    }

    .hero-panel {
        background: radial-gradient(circle at top left, rgba(102, 126, 234, 0.22), transparent 40%),
                    radial-gradient(circle at bottom right, rgba(118, 75, 162, 0.18), transparent 35%),
                    rgba(11, 15, 26, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 28px;
        padding: 40px;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
        margin-bottom: 28px;
    }

    .hero-panel::after {
        content: '';
        position: absolute;
        left: 50%;
        top: 50%;
        width: 520px;
        height: 520px;
        background: radial-gradient(circle, rgba(102,126,234,0.16) 0%, transparent 60%);
        transform: translate(-50%, -50%);
        filter: blur(80px);
        pointer-events: none;
    }

    .hero-panel h1 {
        font-size: clamp(2.4rem, 4vw, 4.2rem);
        margin-bottom: 12px;
        letter-spacing: -1px;
    }

    .hero-panel p {
        color: var(--text-secondary);
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 760px;
        margin: 0 auto;
    }

    .search-panel {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        margin-bottom: 28px;
        backdrop-filter: blur(20px);
    }

    .search-panel h3 {
        margin-bottom: 8px;
        font-size: 1.6rem;
        letter-spacing: -0.4px;
    }

    .search-panel p {
        color: var(--text-secondary);
        margin: 0;
        line-height: 1.6;
    }

    .search-panel .search-meta {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 18px;
    }

    .search-panel .search-meta-icon {
        width: 46px;
        height: 46px;
        border-radius: 16px;
        display: grid;
        place-items: center;
        background: rgba(102, 126, 234, 0.15);
        color: white;
        font-size: 1.4rem;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.22);
    }

    .search-panel input {
        background: rgba(255, 255, 255, 0.08) !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='10' cy='10' r='6' stroke='%23bcbcbc' stroke-width='2' fill='none'/%3E%3Cpath d='M14.5 14.5l5 5' stroke='%23bcbcbc' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: 18px center;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 14px !important;
        padding: 18px 18px 18px 52px !important;
        font-size: 1rem !important;
        color: var(--text-primary) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }

    .search-panel input::placeholder {
        color: rgba(255, 255, 255, 0.55) !important;
        opacity: 1;
        animation: placeholder-fade 3.5s infinite alternate;
    }

    @keyframes placeholder-fade {
        from { opacity: 0.6; }
        to { opacity: 1; }
    }

    .search-panel input::-webkit-textfield-decoration-container {
        padding-left: 48px;
    }

    .chat-container {
        display: grid;
        gap: 18px;
        margin-top: 20px;
    }

    .chat-bubble {
        max-width: 100%;
        border-radius: 24px;
        padding: 18px 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
        backdrop-filter: blur(14px);
        position: relative;
    }

    .chat-bubble.user {
        background: rgba(102, 126, 234, 0.14);
        border-color: rgba(102, 126, 234, 0.25);
        margin-left: auto;
    }

    .chat-bubble.ai {
        background: rgba(255, 255, 255, 0.04);
    }

    .chat-bubble .bubble-meta {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }

    .suggestion-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 999px;
        background: rgba(102, 126, 234, 0.12);
        color: white;
        border: 1px solid rgba(102, 126, 234, 0.25);
        margin: 4px 4px 4px 0;
        cursor: pointer;
        transition: transform 0.2s ease, background 0.2s ease;
    }

    .suggestion-chip:hover {
        background: rgba(102, 126, 234, 0.22);
        transform: translateY(-1px);
    }

    .sidebar-profile {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .avatar-badge {
        width: 60px;
        height: 60px;
        border-radius: 18px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        display: grid;
        place-items: center;
        font-size: 1.8rem;
        margin-bottom: 14px;
        box-shadow: 0 16px 40px rgba(102, 126, 234, 0.2);
    }

    .sidebar-menu {
        display: grid;
        gap: 10px;
        margin-bottom: 18px;
    }

    .sidebar-menu-item {
        display: flex;
        gap: 12px;
        align-items: center;
        padding: 14px 16px;
        border-radius: 18px;
        color: var(--text-primary);
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        cursor: default;
    }

    .sidebar-menu-item:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-1px);
    }

    .sidebar-section {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 18px;
        margin-bottom: 18px;
    }

    .sidebar-section h4 {
        margin: 0 0 8px 0;
        color: var(--text-primary);
    }

    .sidebar-section p {
        margin: 0;
        color: var(--text-secondary);
        font-size: 0.95rem;
    }

    .sidebar-section .upload-note {
        margin-top: 12px;
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.6;
    }

    .button-gradient-glow {
        background: linear-gradient(135deg, #667eea, #764ba2);
        box-shadow: 0 16px 40px rgba(102, 126, 234, 0.24);
        border: none;
        color: white;
    }

    .button-gradient-glow:hover {
        box-shadow: 0 20px 50px rgba(118, 75, 162, 0.32);
    }

    .section-header {
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
    }

    .section-header h3 {
        margin: 0;
        letter-spacing: -0.3px;
    }

    .tab-label {
        padding: 12px 20px;
        border-radius: 999px;
    }
    
    .stTabs [data-baseweb="tab"] {
        min-height: 48px;
        padding: 12px 22px;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(102, 126, 234, 0.95);
        color: white;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.08);
    }

    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat( auto-fit, minmax(220px, 1fr) );
        gap: 20px;
        margin-bottom: 24px;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
        text-align: center;
    }

    .feature-card h4 {
        margin-top: 14px;
        margin-bottom: 8px;
        font-size: 1rem;
    }

    .feature-card p {
        color: var(--text-secondary);
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .metric-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: var(--border-radius-small);
        padding: 16px;
        text-align: center;
        box-shadow: var(--shadow-light);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 8px 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--glass-bg);
        color: var(--text-primary);
        text-align: center;
        border-radius: var(--border-radius-small);
        padding: 8px 12px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(10px);
        box-shadow: var(--shadow-medium);
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None
    if "data_processor" not in st.session_state:
        st.session_state.data_processor = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def display_premium_sidebar():
    """Display premium redesigned sidebar"""
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 20px 0; margin-bottom: 20px;'>
                <div style='background: var(--primary-gradient); border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px; box-shadow: var(--shadow-medium);'>
                    <span style='font-size: 2rem;'>🤖</span>
                </div>
                <h2 style='font-size: 1.4rem; margin: 0 0 5px 0; background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>AI Analytics Pro</h2>
                <p style='color: var(--text-secondary); font-size: 0.85rem; margin: 0; opacity: 0.8;'>Smart Data Intelligence</p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        if st.session_state.logged_in:
            user_name = st.session_state.get('username', 'Analyst')
            st.markdown(f"""
                <div class='sidebar-profile'>
                    <div class='avatar-badge'>{user_name[:1].upper()}</div>
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 1rem; font-weight: 700; color: white;'>{user_name}</div>
                        <div style='font-size: 0.85rem; color: var(--text-secondary);'>Premium User</div>
                    </div>
                    <div style='display: flex; justify-content: space-between; gap: 10px;'>
                        <div style='background: rgba(255,255,255,0.06); border-radius: 14px; padding: 10px 14px; font-size: 0.82rem;'>AI Pro</div>
                        <div style='background: rgba(102,126,234,0.18); border-radius: 14px; padding: 10px 14px; font-size: 0.82rem;'>Active</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("🚪 Logout", help="Logout", use_container_width=True, key="button_1"):
                logout_user()
                st.rerun()

        st.markdown("""
            <div class='sidebar-menu'>
                <div class='sidebar-menu-item'>🤖<span>AI Chat</span></div>
                <div class='sidebar-menu-item'>🔍<span>Data Explorer</span></div>
                <div class='sidebar-menu-item'>📈<span>Statistics</span></div>
                <div class='sidebar-menu-item'>📄<span>Reports</span></div>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
            <div class='sidebar-section'>
                <h4>Upload Your Dataset</h4>
                <p>CSV / XLSX / XLS supported</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📁 Data Management")

        # File upload section
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=['csv', 'xlsx', 'xls'],
            key="file_uploader",
            help="Upload your dataset to get started with AI-powered analytics"
        )

        if uploaded_file is not None:
            @st.cache_data
            def load_data(file):
                processor = DataProcessor()
                if processor.load_file(file):
                    return processor
                return None

            processor = load_data(uploaded_file)
            if processor:
                st.session_state.data_processor = processor
                st.session_state.dataframe = processor.df
                st.session_state.uploaded_filename = uploaded_file.name
                st.success("✅ Dataset loaded successfully!")

        st.divider()

        # Sample data section
        st.markdown("""
            <div class='sidebar-section'>
                <h4>Sample Datasets</h4>
                <p>Load curated data with a single click</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("### 📊 Sample Datasets")
        sample_options = {
            "None": None,
            "📈 Sales Data": "sales",
            "👥 Employee Data": "employees",
            "🌤️ General Data": "general"
        }

        selected_sample = st.selectbox(
            "Choose sample dataset:",
            list(sample_options.keys()),
            key="sample_choice",
            help="Select a sample dataset to explore AI analytics features"
        )

        if selected_sample != "None":
            sample_key = sample_options[selected_sample]
            sample_df = load_sample_data(sample_key)
            st.session_state.data_processor = DataProcessor(sample_df)
            st.session_state.dataframe = sample_df
            st.session_state.uploaded_filename = selected_sample
            st.success(f"✅ Loaded {selected_sample}!")

        # Dataset info
        if st.session_state.dataframe is not None:
            st.divider()
            st.markdown("### 📋 Dataset Info")
            df = st.session_state.dataframe
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
                st.metric("Columns", df.shape[1])
            with col2:
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                numeric_cols = len(get_numeric_columns(df))
                st.metric("Numeric", numeric_cols)


def display_hero_section():
    """Display premium hero section"""
    st.markdown("""
        <div class='hero-panel'>
            <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 24px;'>
                <div style='max-width: 720px;'>
                    <h1>AI Analytics Pro</h1>
                    <p>Smart Data Analysis • Predictions • Insights</p>
                    <p style='margin-top: 18px; color: rgba(255,255,255,0.72);'>Unlock better decisions with a premium analytics workspace designed for modern teams.</p>
                </div>
                <div style='min-width: 180px; padding: 24px; background: rgba(255, 255, 255, 0.05); border-radius: 24px; border: 1px solid rgba(255,255,255,0.08);'>
                    <div style='font-size: 1rem; color: var(--text-secondary); margin-bottom: 10px;'>Realtime Insights</div>
                    <div style='font-size: 2rem; font-weight: 700; background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Live</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def display_kpi_cards():
    """Display premium KPI cards"""
    if st.session_state.dataframe is None:
        return
    
    df = st.session_state.dataframe
    summary = create_dataframe_summary(df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    kpi_data = [
        ("📊", "Rows", f"{summary['rows']:,}", col1),
        ("📋", "Columns", str(summary['columns']), col2),
        ("🔢", "Missing", f"{summary['missing_values']:,}", col3),
        ("🔄", "Duplicates", str(summary['duplicates']), col4),
        ("📈", "Numeric", str(summary['numeric_cols']), col5),
    ]
    
    for icon, label, value, col in kpi_data:
        with col:
            st.markdown(f"""
                <div class='kpi-card'>
                    <div style='font-size: 1.5rem; margin-bottom: 8px;'>{icon}</div>
                    <div class='kpi-value'>{value}</div>
                    <div class='kpi-label'>{label}</div>
                </div>
            """, unsafe_allow_html=True)


def display_ai_search_bar():
    """Display AI-powered search bar at the top"""
    if st.session_state.dataframe is None:
        return

    st.markdown("""
        <div class='search-panel'>
            <div class='search-panel-header'>
                <div class='search-meta-icon'>🔍</div>
                <div>
                    <h3>Ask AI About Your Data</h3>
                    <p>Use natural language to discover trends, forecast behavior, and find powerful recommendations.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])

    with col1:
        ai_query = st.text_input(
            "",
            placeholder="Ask anything about your data...",
            key="global_ai_query",
            label_visibility="collapsed"
        )

    with col2:
        search_button = st.button("Analyze", use_container_width=True, type="primary", key="button_2")

    if search_button and ai_query:
        with st.spinner("🤖 AI is analyzing your query..."):
            df = st.session_state.dataframe
            analyzer = QueryAnalyzer(df)

            result = analyzer.analyze_query(ai_query)

            if result['success']:
                st.success(f"💡 {result['answer']}")

                # Show table if available
                if 'table_data' in result and result['table_data'] is not None:
                    if isinstance(result['table_data'], pd.DataFrame):
                        st.markdown("**📊 Data Table:**")
                        st.dataframe(result['table_data'], use_container_width=True)

                # Show suggested questions
                st.markdown("**💭 Try these related questions:**")
                suggestions = analyzer.get_suggested_questions()[:3]
                cols = st.columns(len(suggestions))
                for i, suggestion in enumerate(suggestions):
                    with cols[i]:
                        if st.button(suggestion, key=f"global_sug_{i}", use_container_width=True):
                            result = analyzer.analyze_query(suggestion)
                            if result['success']:
                                st.success(f"💡 {result['answer']}")
            else:
                st.info(f"🤔 {result['answer']}")

    st.markdown("</div>", unsafe_allow_html=True)


def display_data_explorer():
    """Display data explorer"""
    if st.session_state.dataframe is None:
        return
    
    st.markdown("### 🔍 Data Explorer")
    
    df = st.session_state.dataframe
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Search Data**")
        search_term = st.text_input("Search...", placeholder="Search any value", key="text_input_1")
        if search_term:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            filtered_df = df[mask]
            st.write(f"Found {len(filtered_df)} rows")
            st.dataframe(filtered_df, use_container_width=True, height=300)
    
    with col2:
        st.markdown("**Filter Data**")
        numeric_cols = get_numeric_columns(df)
        categorical_cols = get_categorical_columns(df)
        all_cols = numeric_cols + categorical_cols
        
        if all_cols:
            selected_col = st.selectbox("Column:", all_cols, key="selectbox_1")
            if selected_col in numeric_cols:
                min_val, max_val = st.slider(
                    "Range:",
                    float(df[selected_col].min()),
                    float(df[selected_col].max()),
                    (float(df[selected_col].min()), float(df[selected_col].max())),
                    key="slider_1"
                )
                filtered_df = df[(df[selected_col] >= min_val) & (df[selected_col] <= max_val)]
            else:
                values = df[selected_col].unique().tolist()
                selected_values = st.multiselect("Values:", values, default=values[:3], key="multiselect_1")
                filtered_df = df[df[selected_col].isin(selected_values)]
            
            st.write(f"Showing {len(filtered_df)} rows")


def display_tab_statistics():
    """Display statistics"""
    st.markdown("### 📈 Statistical Analysis")
    df = st.session_state.dataframe
    analyzer = StatisticalAnalysis(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Summary Statistics**")
        st.dataframe(analyzer.get_summary_statistics(), use_container_width=True)
    with col2:
        st.markdown("**Correlation**")
        st.dataframe(analyzer.get_correlation_matrix(), use_container_width=True)


def display_tab_visualizations():
    """Display visualizations"""
    st.markdown("### 📉 Visualizations")
    df = st.session_state.dataframe
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    chart_type = st.selectbox(
        "Chart Type",
        ["Bar Chart", "Pie Chart", "Histogram", "Scatter", "Line Chart"],
        key="selectbox_2"
    )
    
    try:
        if chart_type == "Bar Chart" and categorical_cols and numeric_cols:
            x = st.selectbox("X-axis", categorical_cols, key="selectbox_3")
            y = st.selectbox("Y-axis", numeric_cols, key="selectbox_4")
            fig = px.bar(df.groupby(x)[y].sum().reset_index(), x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Pie Chart" and categorical_cols:
            col = st.selectbox("Column", categorical_cols, key="selectbox_5")
            val_counts = df[col].value_counts().head(10)
            fig = px.pie(values=val_counts.values, names=val_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Histogram" and numeric_cols:
            col = st.selectbox("Column", numeric_cols, key="selectbox_6")
            fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Scatter" and len(numeric_cols) > 1:
            x = st.selectbox("X", numeric_cols, key="selectbox_7")
            y = st.selectbox("Y", numeric_cols, index=(1 if len(numeric_cols) > 1 else 0), key="selectbox_8")
            fig = px.scatter(df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Line Chart":
            x = st.selectbox("X", df.columns, key="selectbox_9")
            y = st.selectbox("Y", numeric_cols, key="selectbox_10")
            fig = px.line(df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")


def display_tab_ml_analysis():
    """Display ML analysis"""
    st.markdown("### 🤖 ML Analysis")
    df = st.session_state.dataframe
    analyzer = MLAnalyzer(df)
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        st.warning("Need 2+ numeric columns")
        return
    
    task = st.selectbox("Task", ["Linear Regression", "Trend Prediction"], key="selectbox_11")
    
    if task == "Linear Regression":
        x = st.selectbox("Feature (X)", numeric_cols, key="selectbox_12")
        y = st.selectbox("Target (Y)", numeric_cols, key="selectbox_13")
        if st.button("Train", key="button_3"):
            result = analyzer.perform_linear_regression(x, y)
            if 'error' not in result:
                c1, c2, c3 = st.columns(3)
                c1.metric("R²", f"{result['r2_score']:.4f}")
                c2.metric("RMSE", f"{result['rmse']:.4f}")
                c3.metric("MAE", f"{result['mae']:.4f}")
                st.success("✅ Done!")
    elif task == "Trend Prediction":
        col = st.selectbox("Column", numeric_cols, key="selectbox_14")
        if st.button("Predict", key="button_4"):
            result = analyzer.predict_trends(col, 5)
            if 'error' not in result:
                st.markdown(f"**Trend:** {result['trend']}")
                st.markdown(f"**Growth:** {result['growth_rate']:.2f}%")


def display_tab_insights():
    """Display insights"""
    st.markdown("### 💡 Insights")
    df = st.session_state.dataframe
    generator = InsightGenerator(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Columns**")
        top = generator.generate_top_columns(top_n=5)
        if top:
            st.dataframe(pd.DataFrame(top), use_container_width=True)
    with col2:
        st.markdown("**Trends**")
        trends = generator.identify_trends()
        if trends:
            st.dataframe(pd.DataFrame(trends[:5]), use_container_width=True)
    
    st.divider()
    st.markdown("**Recommendations:**")
    for i, rec in enumerate(generator.generate_recommendations(), 1):
        st.write(f"{i}. {rec}")


def display_enhanced_data_explorer():
    """Display enhanced data explorer with advanced features"""
    st.markdown("""
        <div class='glass-card'>
            <h3 style='margin: 0 0 20px 0; color: var(--text-primary);'>🔍 Advanced Data Explorer</h3>
            <p style='color: var(--text-secondary); margin: 0;'>Search, filter, and explore your data with powerful tools</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataframe is None:
        return

    df = st.session_state.dataframe
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    # Global search
    st.markdown("### 🔎 Global Search")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_term = st.text_input(
            "Search across all data:",
            placeholder="Enter any value to search...",
            key="global_search",
            help="Search for any value across all columns and rows"
        )
    with search_col2:
        case_sensitive = st.checkbox("Case sensitive", key="case_sensitive")

    if search_term:
        if case_sensitive:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term)).any(axis=1)
        else:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)

        filtered_df = df[mask]
        st.success(f"🔍 Found {len(filtered_df)} matching rows out of {len(df)} total rows")

        if len(filtered_df) > 0:
            st.dataframe(filtered_df, use_container_width=True, height=300)
        else:
            st.info("No matching data found. Try a different search term.")

    st.divider()

    # Advanced filters
    st.markdown("### 🎛️ Advanced Filters")

    # Column selector
    col1, col2 = st.columns(2)
    with col1:
        selected_columns = st.multiselect(
            "Select columns to display:",
            df.columns.tolist(),
            default=df.columns.tolist()[:5] if len(df.columns) > 5 else df.columns.tolist(),
            key="column_selector",
            help="Choose which columns to show in the data table"
        )

    with col2:
        sort_column = st.selectbox(
            "Sort by column:",
            ["None"] + df.columns.tolist(),
            key="sort_column",
            help="Sort the data by any column"
        )

        if sort_column != "None":
            sort_order = st.radio("Sort order:", ["Ascending", "Descending"], key="sort_order", horizontal=True)
            ascending = sort_order == "Ascending"

    # Apply filters
    filtered_df = df.copy()

    # Numeric filters
    if numeric_cols:
        st.markdown("**Numeric Filters:**")
        filter_cols = st.columns(min(3, len(numeric_cols)))

        for i, col in enumerate(numeric_cols[:3]):
            with filter_cols[i]:
                if st.checkbox(f"Filter {col}", key=f"filter_{col}"):
                    min_val, max_val = st.slider(
                        f"{col} range:",
                        float(df[col].min()),
                        float(df[col].max()),
                        (float(df[col].min()), float(df[col].max())),
                        key=f"range_{col}"
                    )
                    filtered_df = filtered_df[
                        (filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)
                    ]

    # Categorical filters
    if categorical_cols:
        st.markdown("**Category Filters:**")
        cat_cols = st.columns(min(2, len(categorical_cols)))

        for i, col in enumerate(categorical_cols[:2]):
            with cat_cols[i]:
                if st.checkbox(f"Filter {col}", key=f"cat_filter_{col}"):
                    values = df[col].unique().tolist()
                    selected_values = st.multiselect(
                        f"Select {col}:",
                        values,
                        default=values,
                        key=f"cat_select_{col}"
                    )
                    if selected_values:
                        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    # Apply sorting
    if sort_column != "None":
        filtered_df = filtered_df.sort_values(sort_column, ascending=ascending)

    # Apply column selection
    if selected_columns:
        filtered_df = filtered_df[selected_columns]

    # Results
    st.divider()
    st.markdown(f"### 📊 Results ({len(filtered_df)} rows)")

    if len(filtered_df) > 0:
        # Pagination
        page_size = st.selectbox("Rows per page:", [10, 25, 50, 100], key="page_size")
        total_pages = (len(filtered_df) - 1) // page_size + 1
        page = st.slider("Page:", 1, total_pages, 1, key="page_slider")

        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))

        st.dataframe(
            filtered_df.iloc[start_idx:end_idx],
            use_container_width=True,
            height=400
        )

        st.info(f"Showing rows {start_idx + 1} to {end_idx} of {len(filtered_df)}")
    else:
        st.warning("No data matches your filters. Try adjusting the criteria.")


def display_enhanced_statistics():
    """Display enhanced statistical analysis with insights"""
    st.markdown("""
        <div class='glass-card'>
            <h3 style='margin: 0 0 20px 0; color: var(--text-primary);'>📈 Advanced Statistical Analysis</h3>
            <p style='color: var(--text-secondary); margin: 0;'>Comprehensive statistical insights with AI-powered explanations</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataframe is None:
        return

    df = st.session_state.dataframe
    analyzer = StatisticalAnalysis(df)
    numeric_cols = get_numeric_columns(df)

    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    stats_df = analyzer.get_summary_statistics()

    if not stats_df.empty:
        # Enhanced display with explanations
        for col in stats_df.index:
            with st.expander(f"📈 {col} - Detailed Analysis", expanded=False):
                col_stats = stats_df.loc[col]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Count", f"{col_stats['Count']:.0f}")
                with col2:
                    st.metric("Mean", f"{col_stats['Mean']:.2f}")
                with col3:
                    st.metric("Std Dev", f"{col_stats['Std Dev']:.2f}")
                with col4:
                    st.metric("Missing", f"{col_stats['Count'] - df[col].count():.0f}")

                # Distribution plot
                if len(df[col].dropna()) > 1:
                    fig = px.histogram(
                        df, x=col, nbins=30,
                        title=f"Distribution of {col}",
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Insights
                st.markdown("**💡 Insights:**")
                if col_stats['Std Dev'] / col_stats['Mean'] > 0.5:
                    st.info("High variability - data points are spread out")
                elif col_stats['Skewness'] > 1:
                    st.warning("Right-skewed distribution - most values are low")
                elif col_stats['Skewness'] < -1:
                    st.warning("Left-skewed distribution - most values are high")
                else:
                    st.success("Normal distribution - balanced data")

    st.divider()

    # Correlation analysis
    st.markdown("### 🔗 Correlation Matrix")
    corr_df = analyzer.get_correlation_matrix()

    if not corr_df.empty:
        # Heatmap
        fig = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Strong correlations
        st.markdown("**💡 Strong Correlations:**")
        strong_corr = []
        for i in range(len(corr_df.columns)):
            for j in range(i+1, len(corr_df.columns)):
                corr_val = corr_df.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append((corr_df.columns[i], corr_df.columns[j], corr_val))

        if strong_corr:
            for col1, col2, corr in strong_corr:
                strength = "Strong Positive" if corr > 0 else "Strong Negative"
                st.info(f"🔗 {col1} ↔ {col2}: {strength} correlation ({corr:.2f})")
        else:
            st.info("No strong correlations found in the data")

    # Data quality insights
    st.divider()
    st.markdown("### 🏥 Data Quality Check")

    quality_cols = st.columns(4)
    with quality_cols[0]:
        st.metric("Total Rows", len(df))
    with quality_cols[1]:
        st.metric("Total Columns", len(df.columns))
    with quality_cols[2]:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with quality_cols[3]:
        dup_rows = df.duplicated().sum()
        st.metric("Duplicate Rows", dup_rows)

    # Quality recommendations
    st.markdown("**💡 Recommendations:**")
    if missing_pct > 10:
        st.warning("⚠️ High missing data percentage. Consider data cleaning.")
    if dup_rows > 0:
        st.warning(f"⚠️ {dup_rows} duplicate rows found. Consider removing duplicates.")
    if len(numeric_cols) < 2:
        st.info("ℹ️ Limited numeric columns for correlation analysis.")


def display_enhanced_visualizations():
    """Display enhanced visualization with auto-suggestions"""
    st.markdown("""
        <div class='glass-card'>
            <h3 style='margin: 0 0 20px 0; color: var(--text-primary);'>📉 Smart Visualizations</h3>
            <p style='color: var(--text-secondary); margin: 0;'>AI-powered charts with automatic insights</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataframe is None:
        return

    df = st.session_state.dataframe
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    # Chart type selector with descriptions
    chart_types = {
        "Bar Chart": "Compare categories or show frequencies",
        "Pie Chart": "Show proportions of categorical data",
        "Histogram": "Display distribution of numeric data",
        "Scatter Plot": "Show relationship between two numeric variables",
        "Line Chart": "Display trends over time or ordered data",
        "Box Plot": "Show distribution statistics and outliers",
        "Heatmap": "Display correlation matrix or 2D data density"
    }

    col1, col2 = st.columns([2, 3])
    with col1:
        chart_type = st.selectbox(
            "Chart Type:",
            list(chart_types.keys()),
            key="chart_type",
            help="Select the type of visualization you want to create"
        )
        st.info(f"💡 {chart_types[chart_type]}")

    with col2:
        st.markdown("**Quick Suggestions:**")
        if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
            if st.button("📊 Top Categories by Value", use_container_width=True, key="button_5"):
                chart_type = "Bar Chart"
        if len(numeric_cols) >= 2:
            if st.button("📈 Correlation Scatter", use_container_width=True, key="button_6"):
                chart_type = "Scatter Plot"
        if len(categorical_cols) >= 1:
            if st.button("🥧 Category Distribution", use_container_width=True, key="button_7"):
                chart_type = "Pie Chart"

    st.divider()

    try:
        if chart_type == "Bar Chart" and categorical_cols and numeric_cols:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("Category (X-axis):", categorical_cols, key="bar_x")
            with col2:
                y_col = st.selectbox("Value (Y-axis):", numeric_cols, key="bar_y")
            with col3:
                agg_func = st.selectbox("Aggregation:", ["sum", "mean", "count"], key="bar_agg")

            if agg_func == "count":
                chart_data = df[x_col].value_counts().head(10).reset_index()
                chart_data.columns = [x_col, 'count']
                fig = px.bar(
                    chart_data, x=x_col, y='count',
                    title=f"Count of {x_col}",
                    color_discrete_sequence=['#667eea']
                )
            else:
                grouped = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                top_10 = grouped.nlargest(10, y_col) if agg_func != "count" else grouped.head(10)
                fig = px.bar(
                    top_10, x=x_col, y=y_col,
                    title=f"{agg_func.title()} of {y_col} by {x_col}",
                    color_discrete_sequence=['#667eea']
                )

            st.plotly_chart(fig, use_container_width=True)

            # Insights
            st.markdown("**💡 Insights:**")
            if agg_func == "sum":
                top_category = grouped.loc[grouped[y_col].idxmax(), x_col]
                st.success(f"🏆 {top_category} has the highest total {y_col}")
            elif agg_func == "mean":
                top_category = grouped.loc[grouped[y_col].idxmax(), x_col]
                st.success(f"📈 {top_category} has the highest average {y_col}")

        elif chart_type == "Pie Chart" and categorical_cols:
            col = st.selectbox("Category Column:", categorical_cols, key="pie_col")
            value_counts = df[col].value_counts().head(10)

            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            st.markdown("**💡 Insights:**")
            top_category = value_counts.index[0]
            top_percentage = (value_counts.values[0] / value_counts.sum()) * 100
            st.success(f"🥧 {top_category} represents {top_percentage:.1f}% of the data")

        elif chart_type == "Histogram" and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                col = st.selectbox("Numeric Column:", numeric_cols, key="hist_col")
            with col2:
                bins = st.slider("Number of bins:", 5, 50, 30, key="hist_bins")

            fig = px.histogram(
                df, x=col, nbins=bins,
                title=f"Distribution of {col}",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            st.markdown("**💡 Insights:**")
            mean_val = df[col].mean()
            median_val = df[col].median()
            st.info(f"📊 Mean: {mean_val:.2f}, Median: {median_val:.2f}")
            if abs(mean_val - median_val) > df[col].std() * 0.5:
                st.warning("⚠️ Distribution may be skewed")

        elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
            with col3:
                color_col = st.selectbox("Color by:", ["None"] + categorical_cols, key="scatter_color")

            if color_col == "None":
                fig = px.scatter(
                    df, x=x_col, y=y_col,
                    title=f"{y_col} vs {x_col}",
                    color_discrete_sequence=['#667eea']
                )
            else:
                fig = px.scatter(
                    df, x=x_col, y=y_col, color=color_col,
                    title=f"{y_col} vs {x_col} by {color_col}"
                )

            st.plotly_chart(fig, use_container_width=True)

            # Correlation insight
            corr = df[x_col].corr(df[y_col])
            st.markdown("**💡 Insights:**")
            if abs(corr) > 0.7:
                strength = "Strong" if abs(corr) > 0.8 else "Moderate"
                direction = "Positive" if corr > 0 else "Negative"
                st.success(f"🔗 {strength} {direction} correlation ({corr:.2f})")
            else:
                st.info(f"🔗 Weak correlation ({corr:.2f})")

        elif chart_type == "Line Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_options = df.columns.tolist()
                x_col = st.selectbox("X-axis (time/category):", x_options, key="line_x")
            with col2:
                y_col = st.selectbox("Y-axis (value):", numeric_cols, key="line_y")

            if df[x_col].dtype in ['object', 'category'] or x_col in categorical_cols:
                # Aggregate by category
                grouped = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.line(
                    grouped, x=x_col, y=y_col,
                    title=f"Average {y_col} by {x_col}",
                    markers=True,
                    color_discrete_sequence=['#667eea']
                )
            else:
                fig = px.line(
                    df, x=x_col, y=y_col,
                    title=f"{y_col} over {x_col}",
                    color_discrete_sequence=['#667eea']
                )

            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot" and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                value_col = st.selectbox("Value Column:", numeric_cols, key="box_value")
            with col2:
                group_col = st.selectbox("Group by:", ["None"] + categorical_cols, key="box_group")

            if group_col == "None":
                fig = px.box(
                    df, y=value_col,
                    title=f"Box Plot of {value_col}",
                    color_discrete_sequence=['#667eea']
                )
            else:
                fig = px.box(
                    df, x=group_col, y=value_col,
                    title=f"{value_col} by {group_col}",
                    color=group_col
                )

            st.plotly_chart(fig, use_container_width=True)

            # Insights
            st.markdown("**💡 Insights:**")
            q1, q3 = df[value_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((df[value_col] < (q1 - 1.5 * iqr)) | (df[value_col] > (q3 + 1.5 * iqr))).sum()
            st.info(f"📦 IQR: {iqr:.2f}, Potential outliers: {outliers}")

        elif chart_type == "Heatmap":
            if len(numeric_cols) >= 2:
                # Correlation heatmap
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation heatmap")

    except Exception as e:
        st.error(f"❌ Error creating chart: {str(e)}")
        st.info("💡 Try selecting different columns or check your data types")


def display_enhanced_ml_analysis():
    """Display enhanced ML analysis with multiple algorithms"""
    st.markdown("""
        <div class='glass-card'>
            <h3 style='margin: 0 0 20px 0; color: var(--text-primary);'>🎯 Advanced ML Analysis</h3>
            <p style='color: var(--text-secondary); margin: 0;'>Predict trends and outcomes with machine learning</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataframe is None:
        return

    df = st.session_state.dataframe
    analyzer = MLAnalyzer(df)
    numeric_cols = get_numeric_columns(df)

    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for ML analysis")
        return

    # Algorithm selector
    ml_algorithms = {
        "Linear Regression": "Predict numeric values using linear relationships",
        "Polynomial Regression": "Predict with curved relationships (degree 2-5)",
        "Random Forest Regression": "Advanced ensemble method for predictions",
        "Trend Analysis": "Analyze and predict future trends"
    }

    col1, col2 = st.columns([2, 3])
    with col1:
        algorithm = st.selectbox(
            "ML Algorithm:",
            list(ml_algorithms.keys()),
            key="ml_algorithm",
            help="Choose the machine learning method for your analysis"
        )
        st.info(f"💡 {ml_algorithms[algorithm]}")

    with col2:
        st.markdown("**Algorithm Comparison:**")
        if st.button("📊 Compare All Models", use_container_width=True, key="button_8"):
            # Compare different algorithms
            results = {}
            x_col = numeric_cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

            # Linear Regression
            lr_result = analyzer.perform_linear_regression(x_col, y_col)
            if 'error' not in lr_result:
                results['Linear Regression'] = lr_result['r2_score']

            # Random Forest
            rf_result = analyzer.perform_random_forest_regression([x_col], y_col)
            if 'error' not in rf_result:
                results['Random Forest'] = rf_result['r2_score']

            if results:
                st.markdown("**Model Performance (R² Score):**")
                for model, score in results.items():
                    st.metric(model, f"{score:.4f}")

    st.divider()

    # Model configuration
    if algorithm in ["Linear Regression", "Polynomial Regression", "Random Forest Regression"]:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Feature Column (X):", numeric_cols, key="ml_x_col")
        with col2:
            y_col = st.selectbox("Target Column (Y):", numeric_cols, key="ml_y_col")
        with col3:
            if algorithm == "Polynomial Regression":
                degree = st.slider("Polynomial Degree:", 2, 5, 2, key="poly_degree")
            elif algorithm == "Random Forest Regression":
                n_estimators = st.slider("Number of Trees:", 10, 200, 100, key="rf_trees")
            else:
                st.write("")

        # Train button
        if st.button("🚀 Train Model", use_container_width=True, type="primary", key="button_9"):
            with st.spinner("🤖 Training machine learning model..."):
                try:
                    if algorithm == "Linear Regression":
                        result = analyzer.perform_linear_regression(x_col, y_col)
                    elif algorithm == "Polynomial Regression":
                        result = analyzer.perform_polynomial_regression(x_col, y_col, degree)
                    elif algorithm == "Random Forest Regression":
                        result = analyzer.perform_random_forest_regression([x_col], y_col, n_estimators)

                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Display results
                        st.success("✅ Model trained successfully!")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{result['r2_score']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{result['rmse']:.4f}")
                        with col3:
                            st.metric("MAE", f"{result['mae']:.4f}")
                        with col4:
                            st.metric("Training Size", f"{len(result['x_values'])} points")

                        # Performance interpretation
                        r2 = result['r2_score']
                        if r2 > 0.8:
                            st.success("🎯 Excellent model performance!")
                        elif r2 > 0.6:
                            st.info("👍 Good model performance")
                        elif r2 > 0.3:
                            st.warning("⚠️ Moderate performance - consider different features")
                        else:
                            st.error("❌ Poor performance - try different algorithm or data")

                        # Visualization
                        if 'predictions' in result and len(result['predictions']) > 0:
                            st.markdown("### 📈 Model Visualization")

                            # Create scatter plot with regression line
                            fig = px.scatter(
                                x=result['x_values'], y=result['actual'],
                                title=f"{algorithm}: {y_col} vs {x_col}",
                                labels={'x': x_col, 'y': y_col}
                            )

                            # Add prediction line
                            if algorithm == "Linear Regression":
                                x_range = np.linspace(min(result['x_values']), max(result['x_values']), 100)
                                y_pred_line = result['intercept'] + result['coefficient'] * x_range
                                fig.add_trace(px.line(x=x_range, y=y_pred_line).data[0])

                            fig.update_traces(marker=dict(color='#667eea'))
                            st.plotly_chart(fig, use_container_width=True)

                        # Feature importance (for Random Forest)
                        if 'feature_importance' in result:
                            st.markdown("### 🎯 Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': [x_col],
                                'Importance': [result['feature_importance']]
                            })
                            st.bar_chart(importance_df.set_index('Feature'))

                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")

    elif algorithm == "Trend Analysis":
        col = st.selectbox("Column to analyze:", numeric_cols, key="trend_col")

        if st.button("📈 Analyze Trend", use_container_width=True, type="primary", key="button_10"):
            with st.spinner("🔍 Analyzing trends..."):
                result = analyzer.predict_trends(col, periods=12)

                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.markdown("### 📊 Trend Analysis Results")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trend Direction", result['trend'])
                    with col2:
                        st.metric("Growth Rate", f"{result['growth_rate']:.2f}%")
                    with col3:
                        st.metric("Confidence", f"{result.get('confidence', 'N/A')}%")

                    # Trend visualization
                    if 'historical' in result and 'predicted' in result:
                        st.markdown("### 📈 Trend Visualization")

                        # Create trend chart
                        fig = px.line(
                            title=f"Trend Analysis: {col}",
                            labels={'value': col, 'index': 'Period'}
                        )

                        # Historical data
                        fig.add_trace(px.scatter(
                            x=range(len(result['historical'])),
                            y=result['historical'],
                            mode='markers',
                            name='Historical'
                        ).data[0])

                        # Trend line
                        fig.add_trace(px.line(
                            x=range(len(result['historical']) + len(result['predicted'])),
                            y=result['historical'] + result['predicted'],
                            name='Trend'
                        ).data[0])

                        st.plotly_chart(fig, use_container_width=True)

                    # Insights
                    st.markdown("**💡 Insights:**")
                    if result['trend'] == 'Increasing':
                        st.success(f"📈 {col} shows an upward trend with {result['growth_rate']:.2f}% growth rate")
                    elif result['trend'] == 'Decreasing':
                        st.warning(f"📉 {col} shows a downward trend with {abs(result['growth_rate']):.2f}% decline rate")
                    else:
                        st.info(f"📊 {col} shows stable trend with minimal change")


def display_enhanced_insights():
    """Display enhanced AI insights with smart recommendations"""
    st.markdown("""
        <div class='glass-card'>
            <h3 style='margin: 0 0 20px 0; color: var(--text-primary);'>💡 AI-Powered Insights</h3>
            <p style='color: var(--text-secondary); margin: 0;'>Intelligent analysis and actionable recommendations</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataframe is None:
        return

    df = st.session_state.dataframe
    generator = InsightGenerator(df)

    # Auto-generated insights
    st.markdown("### 🎯 Key Insights")

    insights_tabs = st.tabs(["📊 Data Overview", "🔍 Patterns", "⚠️ Anomalies", "💡 Recommendations"])

    with insights_tabs[0]:
        st.markdown("**Dataset Summary:**")
        summary = create_dataframe_summary(df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{summary['rows']:,}")
        with col2:
            st.metric("Columns", summary['columns'])
        with col3:
            st.metric("Missing Values", f"{summary['missing_values']:,}")
        with col4:
            st.metric("Duplicates", summary['duplicates'])

        # Data type breakdown
        st.markdown("**Column Types:**")
        numeric_cols = get_numeric_columns(df)
        categorical_cols = get_categorical_columns(df)

        type_col1, type_col2 = st.columns(2)
        with type_col1:
            st.info(f"🔢 Numeric Columns: {len(numeric_cols)}")
            if numeric_cols:
                st.write(", ".join(numeric_cols[:5]) + ("..." if len(numeric_cols) > 5 else ""))

        with type_col2:
            st.info(f"📝 Categorical Columns: {len(categorical_cols)}")
            if categorical_cols:
                st.write(", ".join(categorical_cols[:5]) + ("..." if len(categorical_cols) > 5 else ""))

    with insights_tabs[1]:
        st.markdown("**📈 Data Patterns:**")

        # Top performing columns
        top_cols = generator.generate_top_columns('variance', 5)
        if top_cols:
            st.markdown("**🏆 Top Performing Columns:**")
            for i, col_info in enumerate(top_cols, 1):
                metric_name = col_info['metric'].title()
                score = col_info['score']
                st.success(f"{i}. {col_info['column']} - {metric_name}: {score:.2f}")

        # Trend analysis
        trends = generator.identify_trends()
        if trends:
            st.markdown("**📊 Trend Analysis:**")
            for trend in trends[:5]:
                if trend['trend'] == 'Increasing':
                    st.success(f"📈 {trend['column']}: Increasing trend (+{trend['growth_rate']:.1f}%)")
                elif trend['trend'] == 'Decreasing':
                    st.warning(f"📉 {trend['column']}: Decreasing trend ({trend['growth_rate']:.1f}%)")
                else:
                    st.info(f"📊 {trend['column']}: Stable trend")

    with insights_tabs[2]:
        st.markdown("**⚠️ Data Anomalies:**")

        # Outlier detection
        outliers = generator.identify_outliers_summary()
        if outliers:
            st.markdown("**🚨 Outlier Detection:**")
            for outlier in outliers[:5]:
                st.warning(f"⚠️ {outlier['column']}: {outlier['outlier_count']} outliers detected ({outlier['outlier_percentage']:.1f}% of data)")

        # Missing data analysis
        missing_analysis = generator.analyze_missing_data()
        if missing_analysis:
            st.markdown("**📊 Missing Data Analysis:**")
            for analysis in missing_analysis[:3]:
                severity = "🔴 High" if analysis['percentage'] > 20 else "🟡 Medium" if analysis['percentage'] > 5 else "🟢 Low"
                st.info(f"{severity} {analysis['column']}: {analysis['missing']} missing values ({analysis['percentage']:.1f}%)")

    with insights_tabs[3]:
        st.markdown("**💡 AI Recommendations:**")

        recommendations = generator.generate_recommendations()
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority = "🔴 High" if "critical" in rec.lower() or "missing" in rec.lower() else "🟡 Medium"
                st.info(f"{priority} **Recommendation {i}:** {rec}")
        else:
            st.success("✅ Your data looks good! No major issues detected.")

        # Actionable insights
        st.markdown("**🎯 Actionable Insights:**")

        # Correlation insights
        if len(get_numeric_columns(df)) >= 2:
            corr_insights = generator.generate_correlation_insights()
            if corr_insights:
                st.markdown("**🔗 Correlation Insights:**")
                for insight in corr_insights[:3]:
                    st.info(f"📊 {insight}")

        # Business insights
        business_insights = generator.generate_business_insights()
        if business_insights:
            st.markdown("**💼 Business Insights:**")
            for insight in business_insights[:3]:
                st.success(f"🎯 {insight}")


def display_enhanced_reports():
    """Display enhanced reporting with professional layouts"""
    st.markdown("""
        <div class='glass-card'>
            <h3 style='margin: 0 0 20px 0; color: var(--text-primary);'>📄 Professional Reports</h3>
            <p style='color: var(--text-secondary); margin: 0;'>Generate comprehensive reports in multiple formats</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataframe is None:
        return

    df = st.session_state.dataframe
    report_gen = ReportGenerator(df)
    generator = InsightGenerator(df)

    # Report configuration
    st.markdown("### ⚙️ Report Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        report_title = st.text_input("Report Title:", "AI Analytics Report", key="report_title")
    with col2:
        include_charts = st.checkbox("Include Charts", value=True, key="include_charts")
    with col3:
        include_insights = st.checkbox("Include AI Insights", value=True, key="include_insights")

    st.divider()

    # Report generation options
    st.markdown("### 📋 Generate Reports")

    report_options = st.columns(4)

    with report_options[0]:
        if st.button("📊 Excel Report", use_container_width=True, key="button_11"):
            with st.spinner("Generating Excel report..."):
                excel_data = report_gen.generate_excel_report(report_title)
                if excel_data:
                    st.download_button(
                        "⬇️ Download Excel",
                        excel_data,
                        f"{report_title.replace(' ', '_')}.xlsx",
                        "application/vnd.ms-excel",
                        use_container_width=True
                    )
                    st.success("✅ Excel report generated!")
                else:
                    st.error("❌ Failed to generate Excel report")

    with report_options[1]:
        if st.button("📥 CSV Export", use_container_width=True, key="button_12"):
            with st.spinner("Generating CSV export..."):
                csv_data = report_gen.generate_csv_export()
                if csv_data:
                    st.download_button(
                        "⬇️ Download CSV",
                        csv_data,
                        f"{report_title.replace(' ', '_')}_data.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    st.success("✅ CSV export generated!")
                else:
                    st.error("❌ Failed to generate CSV export")

    with report_options[2]:
        if st.button("📄 Text Summary", use_container_width=True, key="button_13"):
            with st.spinner("Generating text summary..."):
                insights = generator.generate_recommendations() if include_insights else []
                text_report = report_gen.create_summary_report({}, insights)
                if text_report:
                    st.download_button(
                        "⬇️ Download Summary",
                        text_report,
                        f"{report_title.replace(' ', '_')}_summary.txt",
                        "text/plain",
                        use_container_width=True
                    )
                    st.success("✅ Text summary generated!")
                else:
                    st.error("❌ Failed to generate text summary")

    with report_options[3]:
        if st.button("📈 Full Analysis", use_container_width=True, key="button_14"):
            with st.spinner("Generating comprehensive analysis..."):
                # Create comprehensive report
                analysis_data = {
                    'summary': create_dataframe_summary(df),
                    'insights': generator.generate_recommendations(),
                    'trends': generator.identify_trends(),
                    'outliers': generator.identify_outliers_summary()
                }

                # Create a comprehensive text report
                full_report = f"""COMPREHENSIVE DATA ANALYSIS REPORT
{'='*50}

DATASET OVERVIEW:
- Rows: {analysis_data['summary']['rows']:,}
- Columns: {analysis_data['summary']['columns']}
- Missing Values: {analysis_data['summary']['missing_values']:,}
- Duplicates: {analysis_data['summary']['duplicates']}

KEY INSIGHTS:
{chr(10).join(f"• {insight}" for insight in analysis_data['insights'][:10])}

TRENDS IDENTIFIED:
{chr(10).join(f"• {trend['column']}: {trend['trend']} trend" for trend in analysis_data['trends'][:5])}

OUTLIERS DETECTED:
{chr(10).join(f"• {outlier['column']}: {outlier['outlier_count']} outliers ({outlier['outlier_percentage']}%)" for outlier in analysis_data['outliers'][:5])}

Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                if full_report:
                    st.download_button(
                        "⬇️ Download Analysis",
                        full_report,
                        f"{report_title.replace(' ', '_')}_analysis.txt",
                        "text/plain",
                        use_container_width=True
                    )
                    st.success("✅ Comprehensive analysis generated!")
                else:
                    st.error("❌ Failed to generate comprehensive analysis")

    # Report preview
    st.divider()
    st.markdown("### 👀 Report Preview")

    if include_insights:
        st.markdown("**💡 AI Insights Preview:**")
        insights = generator.generate_recommendations()[:3]
        for i, insight in enumerate(insights, 1):
            st.info(f"{i}. {insight}")

    if include_charts and len(get_numeric_columns(df)) >= 2:
        st.markdown("**📊 Sample Chart Preview:**")
        numeric_cols = get_numeric_columns(df)
        fig = px.scatter(
            df, x=numeric_cols[0], y=numeric_cols[1],
            title="Sample Visualization",
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig, use_container_width=True)

    # Report metadata
    st.markdown("**📋 Report Information:**")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Dataset Size", f"{len(df)} rows")
    with info_col2:
        st.metric("Generated", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    with info_col3:
        st.metric("Format", "Multiple (Excel, CSV, Text)")


def display_enhanced_ai_chat():
    """Display enhanced AI chat with history and better UX"""
    st.markdown("""
        <div class='glass-card'>
            <h3 style='margin: 0 0 20px 0; color: var(--text-primary);'>🤖 AI Data Assistant</h3>
            <p style='color: var(--text-secondary); margin: 0;'>Ask intelligent questions about your data in natural language</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataframe is None:
        return

    df = st.session_state.dataframe
    analyzer = QueryAnalyzer(df)

    # Chat history section
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        st.markdown("### 💬 Recent Conversations")
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for query, response in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"""
                <div class='chat-bubble user'>
                    <div class='bubble-meta'>You</div>
                    <div>{query}</div>
                </div>
                <div class='chat-bubble ai'>
                    <div class='bubble-meta'>AI Response</div>
                    <div>{response}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()

    # Main query input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Ask your question:",
            placeholder="e.g., 'What are the top 5 products by sales?', 'Show me customer trends', 'Find data outliers'",
            key="enhanced_ai_query",
            help="Ask anything about your data - the AI will understand and provide insights"
        )
    with col2:
        analyze_btn = st.button("Analyze", use_container_width=True, type="primary", key="button_15")

    if analyze_btn and query:
        with st.spinner("🤖 Analyzing your query with AI..."):
            result = analyzer.analyze_query(query)

            if result['success']:
                # Add to chat history
                st.session_state.chat_history.append((query, result['answer']))

                # Display result as AI chat bubble
                st.markdown("""
                    <div class='chat-container' style='margin-top: 20px;'>
                        <div class='chat-bubble user'>
                            <div class='bubble-meta'>You</div>
                            <div>{query}</div>
                        </div>
                        <div class='chat-bubble ai'>
                            <div class='bubble-meta'>AI Response</div>
                            <div>{result['answer']}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Show data table if available
                if 'table_data' in result and result['table_data'] is not None:
                    if isinstance(result['table_data'], pd.DataFrame):
                        st.markdown("**📊 Supporting Data:**")
                        st.dataframe(result['table_data'], use_container_width=True)

                # Show visualization if applicable
                if 'chart_data' in result and result['chart_data'] is not None:
                    st.markdown("**📈 Visual Insight:**")
                    # Add chart display logic here

            else:
                st.info(f"🤔 {result['answer']}")

    # Suggested questions
    st.markdown("### 💭 Suggested Questions")
    suggestions = analyzer.get_suggested_questions()

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions[:6]):
        with cols[i % 2]:
            if st.button(
                suggestion,
                key=f"enhanced_sug_{i}",
                use_container_width=True,
                help=f"Click to ask: {suggestion}"
            ):
                with st.spinner("🤖 Processing..."):
                    result = analyzer.analyze_query(suggestion)
                    if result['success']:
                        st.session_state.chat_history.append((suggestion, result['answer']))
                        st.success(f"💡 {result['answer']}")
                        if 'table_data' in result and result['table_data'] is not None:
                            st.dataframe(result['table_data'], use_container_width=True)


def display_tab_reports():
    """Display reports"""
    st.markdown("### 📄 Reports")
    df = st.session_state.dataframe
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 Excel", use_container_width=True, key="button_16"):
            report_gen = ReportGenerator(df)
            excel_bytes = report_gen.generate_excel_report("report.xlsx")
            if excel_bytes:
                st.download_button("⬇️", excel_bytes, "report.xlsx", "application/vnd.ms-excel")
    with c2:
        if st.button("📥 CSV", use_container_width=True, key="button_17"):
            report_gen = ReportGenerator(df)
            csv_bytes = report_gen.generate_csv_export()
            if csv_bytes:
                st.download_button("⬇️", csv_bytes, "data.csv", "text/csv")
    with c3:
        if st.button("📝 Text", use_container_width=True, key="button_18"):
            report_gen = ReportGenerator(df)
            generator = InsightGenerator(df)
            text = report_gen.create_summary_report({}, generator.generate_recommendations())
            st.download_button("⬇️", text, "report.txt", "text/plain")


def display_premium_dashboard():
    """Display premium dashboard with enhanced layout"""
    display_premium_sidebar()
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    if st.session_state.dataframe is None:
        # Enhanced welcome screen
        st.markdown("""
            <div style='text-align: center; padding: 60px 20px;'>
                <div style='background: var(--primary-gradient); border-radius: 50%; width: 120px; height: 120px; display: flex; align-items: center; justify-content: center; margin: 0 auto 30px; box-shadow: var(--shadow-heavy);'>
                    <span style='font-size: 4rem;'>🤖</span>
                </div>
                <h1 style='font-size: 3.5rem; margin: 0 0 20px 0; background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>AI Analytics Pro</h1>
                <p style='font-size: 1.4rem; color: var(--text-secondary); margin: 0 0 40px 0; max-width: 600px; margin-left: auto; margin-right: auto;'>Transform your data into actionable insights with AI-powered analytics</p>
            </div>
        """, unsafe_allow_html=True)

        # Feature highlights
        col1, col2, col3, col4 = st.columns(4)
        features = [
            ("🤖", "AI Chat", "Ask questions in natural language"),
            ("📊", "Smart Visualizations", "Auto-generated charts & graphs"),
            ("🎯", "ML Predictions", "Predict trends & outcomes"),
            ("💡", "Auto Insights", "AI-generated recommendations")
        ]

        for i, (icon, title, desc) in enumerate(features):
            with [col1, col2, col3, col4][i]:
                st.markdown(f"""
                    <div class='glass-card' style='text-align: center; padding: 20px;'>
                        <div style='font-size: 2rem; margin-bottom: 10px;'>{icon}</div>
                        <div style='font-weight: 600; color: var(--text-primary); margin-bottom: 5px;'>{title}</div>
                        <div style='font-size: 0.85rem; color: var(--text-secondary);'>{desc}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("""
            <div style='text-align: center; margin-top: 40px; padding: 30px; background: var(--glass-bg); border-radius: var(--border-radius); border: 1px solid var(--glass-border); backdrop-filter: blur(20px);'>
                <h3 style='color: var(--text-primary); margin: 0 0 20px 0;'>🚀 Get Started</h3>
                <p style='color: var(--text-secondary); margin: 0 0 20px 0;'>Upload your data or try our sample datasets to explore AI-powered analytics</p>
                <div style='display: flex; justify-content: center; gap: 20px;'>
                    <div style='background: var(--primary-gradient); color: white; padding: 12px 24px; border-radius: 8px; font-weight: 600;'>📤 Upload Data</div>
                    <div style='background: rgba(255, 255, 255, 0.1); color: var(--text-primary); padding: 12px 24px; border-radius: 8px; font-weight: 600; border: 1px solid var(--glass-border);'>📊 Try Samples</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        return

    # Main dashboard with data
    display_hero_section()
    display_ai_search_bar()
    display_kpi_cards()
    st.divider()

    # Enhanced tabs with better styling
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🤖 AI Chat", "🔍 Data Explorer", "📈 Statistics", "📉 Visualizations", "🎯 ML Analysis", "💡 Insights", "📄 Reports"
    ])

    with tab1:
        display_enhanced_ai_chat()
    with tab2:
        display_enhanced_data_explorer()
    with tab3:
        display_enhanced_statistics()
    with tab4:
        display_enhanced_visualizations()
    with tab5:
        display_enhanced_ml_analysis()
    with tab6:
        display_enhanced_insights()
    with tab7:
        display_enhanced_reports()

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main function"""
    initialize_session_state()
    
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🔐 AI Analytics Pro")
            st.divider()
            username = st.text_input("Username", placeholder="admin", key="user")
            password = st.text_input("Password", type="password", placeholder="admin123", key="pass")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔓 Login", use_container_width=True, key="button_19"):
                    if login_user(username, password):
                        st.success("✅ Welcome!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid!")
            st.divider()
            with st.expander("ℹ️ Demo Accounts"):
                st.info("**admin** / admin123\n**user** / password123\n**analyst** / analyst@2024")
    else:
        display_premium_dashboard()


if __name__ == "__main__":
    main()