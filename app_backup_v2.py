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
        --glass-bg: rgba(15, 23, 42, 0.8);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    * {
        transition: all 0.3s ease;
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

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .glass-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }

    .kpi-card {
        background: var(--glass-bg);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }

    .kpi-card:hover {
        border-color: rgba(102, 126, 234, 0.6);
        transform: scale(1.02);
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 10px 0;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.03);
        padding: 10px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #cbd5e1;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary-gradient);
        color: white;
    }

    .stSuccess {
        background: rgba(34, 202, 102, 0.1);
        border-left: 4px solid #22ca66;
        border-radius: 8px;
        padding: 15px;
    }

    .stError {
        background: rgba(255, 85, 85, 0.1);
        border-left: 4px solid #ff5555;
        border-radius: 8px;
        padding: 15px;
    }

    .stInfo {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 15px;
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
            <div style='text-align: center; padding: 20px 0;'>
                <h1 style='font-size: 2rem; margin: 0;'>🤖</h1>
                <h2 style='font-size: 1.2rem; margin: 5px 0;'>AI Analytics</h2>
                <p style='color: #cbd5e1; font-size: 0.85rem; margin: 0;'>Pro Dashboard</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        if st.session_state.logged_in:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"👤 **{st.session_state.username}**")
            with col2:
                if st.button("🚪", help="Logout", use_container_width=False):
                    logout_user()
                    st.rerun()
        
        st.divider()
        
        st.markdown("### 📁 Your Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=['csv', 'xlsx', 'xls'],
            key="file_uploader"
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
                st.success("✅ Loaded!")
        
        st.divider()
        
        st.markdown("### 📊 Samples")
        sample_choice = st.selectbox(
            "Load sample",
            ["None", "Sales Data", "Employee Data", "General Data"],
            key="sample_choice"
        )
        
        if sample_choice != "None":
            sample_map = {
                "Sales Data": "sales",
                "Employee Data": "employees",
                "General Data": "general"
            }
            sample_df = load_sample_data(sample_map[sample_choice])
            st.session_state.data_processor = DataProcessor(sample_df)
            st.session_state.dataframe = sample_df
            st.session_state.uploaded_filename = f"{sample_choice}"
            st.success(f"✅ Loaded!")


def display_hero_section():
    """Display premium hero section"""
    st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border: 2px solid rgba(102, 126, 234, 0.3);
            border-radius: 20px;
            padding: 40px;
            margin: 20px 0;
            text-align: center;
        '>
            <h1 style='font-size: 3rem; margin: 0 0 10px 0;'>🤖 AI Analytics Pro</h1>
            <p style='font-size: 1.3rem; color: #cbd5e1; margin: 0;'>Smart Data Analysis • Predictions • Insights</p>
            <p style='font-size: 0.95rem; color: #94a3b8; margin: 15px 0 0 0;'>Enterprise Analytics With AI Power</p>
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


def display_ai_query_section():
    """Display AI Query feature"""
    if st.session_state.dataframe is None:
        return
    
    st.markdown("### 🤖 Ask About Your Data")
    
    df = st.session_state.dataframe
    analyzer = QueryAnalyzer(df)
    
    query = st.text_input(
        "💬 Ask anything about your data...",
        placeholder="e.g., Which product has highest sales? What's the average?",
        key="ai_query"
    )
    
    if st.button("🔍 Analyze", use_container_width=False):
        if query:
            result = analyzer.analyze_query(query)
            if result['success']:
                st.success(result['answer'])
                if 'table_data' in result and result['table_data'] is not None:
                    if isinstance(result['table_data'], pd.DataFrame):
                        st.dataframe(result['table_data'], use_container_width=True)
            else:
                st.info(result['answer'])
    
    st.markdown("**Suggested:**")
    suggestions = analyzer.get_suggested_questions()
    for suggestion in suggestions:
        if st.button(suggestion, key=f"sug_{suggestion}", use_container_width=True):
            result = analyzer.analyze_query(suggestion)
            if result['success']:
                st.success(result['answer'])


def display_data_explorer():
    """Display data explorer"""
    if st.session_state.dataframe is None:
        return
    
    st.markdown("### 🔍 Data Explorer")
    
    df = st.session_state.dataframe
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Search Data**")
        search_term = st.text_input("Search...", placeholder="Search any value")
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
            selected_col = st.selectbox("Column:", all_cols)
            if selected_col in numeric_cols:
                min_val, max_val = st.slider(
                    "Range:",
                    float(df[selected_col].min()),
                    float(df[selected_col].max()),
                    (float(df[selected_col].min()), float(df[selected_col].max()))
                )
                filtered_df = df[(df[selected_col] >= min_val) & (df[selected_col] <= max_val)]
            else:
                values = df[selected_col].unique().tolist()
                selected_values = st.multiselect("Values:", values, default=values[:3])
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
        ["Bar Chart", "Pie Chart", "Histogram", "Scatter", "Line Chart"]
    )
    
    try:
        if chart_type == "Bar Chart" and categorical_cols and numeric_cols:
            x = st.selectbox("X-axis", categorical_cols)
            y = st.selectbox("Y-axis", numeric_cols)
            fig = px.bar(df.groupby(x)[y].sum().reset_index(), x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Pie Chart" and categorical_cols:
            col = st.selectbox("Column", categorical_cols)
            val_counts = df[col].value_counts().head(10)
            fig = px.pie(values=val_counts.values, names=val_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Histogram" and numeric_cols:
            col = st.selectbox("Column", numeric_cols)
            fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Scatter" and len(numeric_cols) > 1:
            x = st.selectbox("X", numeric_cols)
            y = st.selectbox("Y", numeric_cols, index=(1 if len(numeric_cols) > 1 else 0))
            fig = px.scatter(df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Line Chart":
            x = st.selectbox("X", df.columns)
            y = st.selectbox("Y", numeric_cols)
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
    
    task = st.selectbox("Task", ["Linear Regression", "Trend Prediction"])
    
    if task == "Linear Regression":
        x = st.selectbox("Feature (X)", numeric_cols)
        y = st.selectbox("Target (Y)", numeric_cols)
        if st.button("Train"):
            result = analyzer.perform_linear_regression(x, y)
            if 'error' not in result:
                c1, c2, c3 = st.columns(3)
                c1.metric("R²", f"{result['r2_score']:.4f}")
                c2.metric("RMSE", f"{result['rmse']:.4f}")
                c3.metric("MAE", f"{result['mae']:.4f}")
                st.success("✅ Done!")
    elif task == "Trend Prediction":
        col = st.selectbox("Column", numeric_cols)
        if st.button("Predict"):
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


def display_tab_reports():
    """Display reports"""
    st.markdown("### 📄 Reports")
    df = st.session_state.dataframe
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 Excel", use_container_width=True):
            report_gen = ReportGenerator(df)
            excel_bytes = report_gen.generate_excel_report("report.xlsx")
            if excel_bytes:
                st.download_button("⬇️", excel_bytes, "report.xlsx", "application/vnd.ms-excel")
    with c2:
        if st.button("📥 CSV", use_container_width=True):
            report_gen = ReportGenerator(df)
            csv_bytes = report_gen.generate_csv_export()
            if csv_bytes:
                st.download_button("⬇️", csv_bytes, "data.csv", "text/csv")
    with c3:
        if st.button("📝 Text", use_container_width=True):
            report_gen = ReportGenerator(df)
            generator = InsightGenerator(df)
            text = report_gen.create_summary_report({}, generator.generate_recommendations())
            st.download_button("⬇️", text, "report.txt", "text/plain")


def display_premium_dashboard():
    """Display premium dashboard"""
    display_premium_sidebar()
    
    if st.session_state.dataframe is None:
        display_hero_section()
        st.info("""
        👋 **Welcome to AI Analytics Pro!**
        
        **Quick Start:**
        1. Upload data in sidebar
        2. Or load sample dataset
        3. Explore with tabs
        
        **Features:** 🤖 AI Query • 📊 Data Explore • 📈 Analysis • 📉 Charts • 🎯 ML • 💡 Insights
        """)
        return
    
    display_kpi_cards()
    st.divider()
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🤖 AI Chat", "🔍 Explore", "📈 Stats", "📉 Charts", "🎯 ML", "💡 Insights", "📄 Reports"
    ])
    
    with tab1:
        display_ai_query_section()
    with tab2:
        display_data_explorer()
    with tab3:
        display_tab_statistics()
    with tab4:
        display_tab_visualizations()
    with tab5:
        display_tab_ml_analysis()
    with tab6:
        display_tab_insights()
    with tab7:
        display_tab_reports()


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
                if st.button("🔓 Login", use_container_width=True):
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
