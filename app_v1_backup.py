"""
AI-Based Data Analytics Agent
A professional data analysis system with ML insights, visualizations, and report generation
Author: AI Assistant
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.auth import (
    display_login_form, login_user, logout_user, 
    check_login_status, display_logout_button
)
from modules.data_processor import DataProcessor
from modules.statistical_analysis import StatisticalAnalysis
from modules.visualization import Visualizer
from modules.ml_analysis import MLAnalyzer
from modules.insight_generator import InsightGenerator
from modules.report_generator import ReportGenerator
from modules.utils import (
    load_sample_data, get_numeric_columns, get_categorical_columns,
    create_dataframe_summary, validate_dataframe, get_time_based_greeting,
    display_loading_animation
)


# Page configuration
st.set_page_config(
    page_title="AI Analytics Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
st.markdown("""
    <style>
    /* Dark theme with gradient */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --background-color: #0f0f1e;
        --surface-color: #1a1a2e;
    }
    
    /* Main containers */
    .main {
        background-color: var(--background-color);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Metrics cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        background-color: transparent;
        color: #667eea;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stTabBarContainerButton"] {
        background-color: transparent;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 2px solid #667eea;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .stSuccess {
        background-color: rgba(34, 202, 102, 0.1);
        border-left: 4px solid #22ca66;
    }
    
    .stError {
        background-color: rgba(255, 85, 85, 0.1);
        border-left: 4px solid #ff5555;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
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


def display_header():
    """Display application header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center; color: #667eea;'>🤖 AI Analytics Agent</h1>", 
                   unsafe_allow_html=True)
        st.markdown("""
            <p style='text-align: center; color: #999; font-size: 14px;'>
            Intelligent Data Analysis • ML Predictions • Business Insights
            </p>
        """, unsafe_allow_html=True)


def display_login_page():
    """Display login page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔐 Login")
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            username = st.text_input(
                "Username",
                placeholder="admin",
                key="login_input_username"
            )
        
        with col_right:
            password = st.text_input(
                "Password",
                type="password",
                placeholder="admin123",
                key="login_input_password"
            )
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn2:
            if st.button("🔓 Login", use_container_width=True):
                if login_user(username, password):
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password!")
        
        st.markdown("---")
        with st.expander("ℹ️ Demo Credentials"):
            st.info("""
            **Test Accounts:**
            - **Username:** admin | **Password:** admin123
            - **Username:** user | **Password:** password123
            - **Username:** analyst | **Password:** analyst@2024
            """)


def display_dashboard():
    """Display main dashboard"""
    display_logout_button()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 📊 Data Controls")
        
        # File upload section
        st.markdown("#### 📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
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
                st.success(f"✅ Loaded: {uploaded_file.name}")
        
        # Sample data
        st.markdown("#### 📌 Load Sample Data")
        sample_choice = st.selectbox(
            "Select sample dataset",
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
            st.session_state.uploaded_filename = f"{sample_choice} (Sample)"
            st.success(f"✅ Loaded: {sample_choice}")
    
    # Main content area
    if st.session_state.dataframe is None:
        display_header()
        st.info("""
        👋 Welcome to AI Analytics Agent!
        
        **Get Started:**
        1. Upload a CSV or Excel file using the sidebar
        2. Or select a sample dataset
        3. Navigate through the tabs for analysis
        
        **Features:**
        - 📊 Data Preview & Summary
        - 🧹 Data Cleaning & Preprocessing
        - 📈 Statistical Analysis
        - 📉 Interactive Visualizations
        - 🤖 ML Analysis & Predictions
        - 💡 Auto-Generated Insights
        - 📄 Report Generation
        """)
    else:
        display_header()
        
        # Display current file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 File", st.session_state.uploaded_filename or "None")
        with col2:
            st.metric("📊 Rows", len(st.session_state.dataframe))
        with col3:
            st.metric("📋 Columns", len(st.session_state.dataframe.columns))
        
        st.markdown("---")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview",
            "🧹 Cleaning",
            "📈 Statistics",
            "📉 Visualizations",
            "🤖 ML Analysis",
            "💡 Insights",
            "📄 Reports"
        ])
        
        with tab1:
            display_tab_overview()
        
        with tab2:
            display_tab_cleaning()
        
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


def display_tab_overview():
    """Display data overview tab"""
    st.markdown("### 📊 Dataset Overview")
    
    df = st.session_state.dataframe
    
    # Summary statistics
    summary = create_dataframe_summary(df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Rows", f"{summary['rows']:,}")
    with col2:
        st.metric("Total Columns", summary['columns'])
    with col3:
        st.metric("Numeric Cols", summary['numeric_cols'])
    with col4:
        st.metric("Categorical Cols", summary['categorical_cols'])
    with col5:
        st.metric("Missing Values", f"{summary['missing_values']:,}")
    
    st.markdown("---")
    
    # Data preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Column Information")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Missing': df.isnull().sum()
        })
        st.dataframe(info_df, use_container_width=True)
    
    st.markdown("---")
    
    # Data sample download
    st.markdown("#### 📥 Export Data View")
    csv = df.head(100).to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV (First 100 rows)",
        data=csv,
        file_name="data_preview.csv",
        mime="text/csv"
    )


def display_tab_cleaning():
    """Display data cleaning tab"""
    st.markdown("### 🧹 Data Preprocessing")
    
    df = st.session_state.dataframe
    processor = st.session_state.data_processor
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Values Handling")
        
        with st.form("missing_values_form"):
            strategy = st.selectbox(
                "Select strategy for missing values",
                ["mean", "median"]
            )
            
            if st.form_submit_button("🧹 Handle Missing Values"):
                with st.spinner("Processing..."):
                    processor.handle_missing_values(numeric_strategy=strategy)
                    st.session_state.dataframe = processor.df
                    st.success("✅ Missing values handled!")
                    st.rerun()
    
    with col2:
        st.markdown("#### Remove Duplicates")
        
        duplicate_count = df.duplicated().sum()
        st.info(f"Found {duplicate_count} duplicate rows")
        
        if st.button("🔄 Remove Duplicates", use_container_width=True):
            with st.spinner("Removing duplicates..."):
                processor.remove_duplicates()
                st.session_state.dataframe = processor.df
                st.success("✅ Duplicates removed!")
                st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Remove Outliers")
        
        with st.form("outliers_form"):
            method = st.selectbox("Outlier detection method", ["iqr", "zscore"])
            
            if st.form_submit_button("🎯 Remove Outliers"):
                with st.spinner("Detecting outliers..."):
                    processor.remove_outliers(method=method)
                    st.session_state.dataframe = processor.df
                    st.success("✅ Outliers removed!")
                    st.rerun()
    
    with col2:
        st.markdown("#### Encode Categorical Data")
        
        categorical_cols = get_categorical_columns(df)
        
        if categorical_cols:
            if st.button("🔤 Encode Categorical Columns", use_container_width=True):
                with st.spinner("Encoding..."):
                    processor.encode_categorical(categorical_cols)
                    st.session_state.dataframe = processor.df
                    st.success("✅ Categorical data encoded!")
                    st.rerun()
        else:
            st.info("No categorical columns found")
    
    st.markdown("---")
    
    # Processing log
    st.markdown("#### 📝 Processing Log")
    log = processor.get_processing_log()
    
    if log:
        for idx, entry in enumerate(log, 1):
            st.write(f"{idx}. {entry}")
    else:
        st.info("No processing operations performed yet")
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔃 Reset to Original Data", use_container_width=True):
        processor.reset_to_original()
        st.session_state.dataframe = processor.df
        st.success("✅ Data reset!")
        st.rerun()


def display_tab_statistics():
    """Display statistical analysis tab"""
    st.markdown("### 📈 Statistical Analysis")
    
    df = st.session_state.dataframe
    analyzer = StatisticalAnalysis(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Summary Statistics")
        summary_stats = analyzer.get_summary_statistics()
        st.dataframe(summary_stats, use_container_width=True)
    
    with col2:
        st.markdown("#### Correlation Matrix")
        correlation = analyzer.get_correlation_matrix()
        st.dataframe(correlation, use_container_width=True)
    
    st.markdown("---")
    
    # Column-specific statistics
    st.markdown("#### 📊 Column Analysis")
    
    col = st.selectbox(
        "Select column for detailed analysis",
        df.columns,
        key="stat_column_select"
    )
    
    if col:
        col_stats = analyzer.get_column_statistics(col)
        
        if col_stats:
            stats_df = pd.DataFrame(list(col_stats.items()), columns=['Metric', 'Value'])
            st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("---")
    
    # Missing data analysis
    st.markdown("#### Missing Data Analysis")
    
    missing_info = analyzer.identify_missing_pattern()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Missing", missing_info['total_missing'])
    with col2:
        st.metric("Missing %", f"{missing_info['missing_percentage']:.2f}%")
    with col3:
        st.metric("Affected Columns", len(missing_info['columns_with_missing']))
    
    if missing_info['columns_with_missing']:
        st.dataframe(
            pd.DataFrame(missing_info['columns_with_missing']).T,
            use_container_width=True
        )


def display_tab_visualizations():
    """Display visualizations tab"""
    st.markdown("### 📉 Interactive Visualizations")
    
    df = st.session_state.dataframe
    visualizer = Visualizer(df)
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    st.markdown("#### Select Chart Type")
    
    chart_type = st.selectbox(
        "Choose visualization",
        [
            "Bar Chart",
            "Line Chart",
            "Pie Chart",
            "Histogram",
            "Scatter Plot",
            "Heatmap",
            "Box Plot",
            "Distribution Plot"
        ]
    )
    
    st.markdown("---")
    
    try:
        if chart_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", categorical_cols or numeric_cols, key="bar_x")
            with col2:
                y_col = st.selectbox("Y-axis (optional)", ["None"] + numeric_cols, key="bar_y")
            
            if x_col:
                fig = visualizer.create_bar_chart(x_col, y_col if y_col != "None" else None)
                st.pyplot(fig)
        
        elif chart_type == "Line Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols + categorical_cols, key="line_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, key="line_y")
            
            if x_col and y_col:
                fig = visualizer.create_line_chart(x_col, y_col)
                st.pyplot(fig)
        
        elif chart_type == "Pie Chart":
            col = st.selectbox("Select column", categorical_cols or numeric_cols, key="pie_col")
            if col:
                fig = visualizer.create_pie_chart(col)
                st.pyplot(fig)
        
        elif chart_type == "Histogram":
            col = st.selectbox("Select numeric column", numeric_cols, key="hist_col")
            bins = st.slider("Number of bins", 5, 50, 30, key="hist_bins")
            if col:
                fig = visualizer.create_histogram(col, bins=bins)
                st.pyplot(fig)
        
        elif chart_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
            
            if x_col and y_col:
                fig = visualizer.create_scatter_plot(x_col, y_col)
                st.pyplot(fig)
        
        elif chart_type == "Heatmap":
            fig = visualizer.create_heatmap()
            st.pyplot(fig)
        
        elif chart_type == "Box Plot":
            selected_cols = st.multiselect(
                "Select columns",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            if selected_cols:
                fig = visualizer.create_box_plot(selected_cols)
                st.pyplot(fig)
        
        elif chart_type == "Distribution Plot":
            col = st.selectbox("Select numeric column", numeric_cols, key="dist_col")
            if col:
                fig = visualizer.create_distribution_plot(col)
                st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")


def display_tab_ml_analysis():
    """Display ML analysis tab"""
    st.markdown("### 🤖 Machine Learning Analysis")
    
    df = st.session_state.dataframe
    analyzer = MLAnalyzer(df)
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols or len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for ML analysis")
        return
    
    ml_task = st.selectbox(
        "Select ML Task",
        [
            "Linear Regression",
            "Polynomial Regression",
            "Random Forest Regression",
            "Trend Prediction"
        ]
    )
    
    st.markdown("---")
    
    try:
        if ml_task == "Linear Regression":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Feature (X)", numeric_cols, key="lr_x")
            with col2:
                y_col = st.selectbox("Target (Y)", numeric_cols, key="lr_y")
            
            if st.button("🎯 Train Model"):
                with st.spinner("Training..."):
                    results = analyzer.perform_linear_regression(x_col, y_col)
                    
                    if 'error' not in results:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{results['r2_score']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{results['rmse']:.4f}")
                        with col3:
                            st.metric("MAE", f"{results['mae']:.4f}")
                        
                        st.success("✅ Model trained successfully!")
                    else:
                        st.error(f"Error: {results['error']}")
        
        elif ml_task == "Polynomial Regression":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("Feature (X)", numeric_cols, key="poly_x")
            with col2:
                y_col = st.selectbox("Target (Y)", numeric_cols, key="poly_y")
            with col3:
                degree = st.slider("Polynomial Degree", 2, 5, 2)
            
            if st.button("🎯 Train Model"):
                with st.spinner("Training..."):
                    results = analyzer.perform_polynomial_regression(x_col, y_col, degree)
                    
                    if 'error' not in results:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{results['r2_score']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{results['rmse']:.4f}")
                        with col3:
                            st.metric("MAE", f"{results['mae']:.4f}")
                        
                        st.success("✅ Model trained successfully!")
                    else:
                        st.error(f"Error: {results['error']}")
        
        elif ml_task == "Random Forest Regression":
            selected_features = st.multiselect(
                "Select features",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            target = st.selectbox("Select target", numeric_cols, key="rf_target")
            n_estimators = st.slider("Number of trees", 10, 200, 100)
            
            if st.button("🎯 Train Model"):
                with st.spinner("Training..."):
                    results = analyzer.perform_random_forest_regression(
                        selected_features, target, n_estimators
                    )
                    
                    if 'error' not in results:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{results['r2_score']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{results['rmse']:.4f}")
                        with col3:
                            st.metric("MAE", f"{results['mae']:.4f}")
                        
                        st.markdown("#### Feature Importance")
                        st.bar_chart(
                            results['feature_importance'].set_index('feature')
                        )
                        
                        st.success("✅ Model trained successfully!")
                    else:
                        st.error(f"Error: {results['error']}")
        
        elif ml_task == "Trend Prediction":
            col = st.selectbox("Select column for trend analysis", numeric_cols, key="trend_col")
            periods = st.slider("Periods to predict", 1, 20, 5)
            
            if st.button("📊 Analyze Trend"):
                with st.spinner("Analyzing..."):
                    results = analyzer.predict_trends(col, periods)
                    
                    if 'error' not in results:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Trend", results['trend'])
                        with col2:
                            st.metric("Growth Rate", f"{results['growth_rate']:.2f}%")
                        with col3:
                            st.metric("Slope", f"{results['slope']:.4f}")
                        with col4:
                            st.metric("Periods", periods)
                        
                        st.success("✅ Trend analysis complete!")
                    else:
                        st.error(f"Error: {results['error']}")
    
    except Exception as e:
        st.error(f"Error in ML analysis: {str(e)}")


def display_tab_insights():
    """Display insights and recommendations tab"""
    st.markdown("### 💡 AI-Generated Insights")
    
    df = st.session_state.dataframe
    generator = InsightGenerator(df)
    
    # Top performing columns
    st.markdown("#### 🏆 Top Performing Columns")
    
    top_cols = generator.generate_top_columns(metric='variance', top_n=5)
    
    if top_cols:
        top_df = pd.DataFrame(top_cols)
        st.dataframe(top_df, use_container_width=True)
    else:
        st.info("No numeric columns available for analysis")
    
    st.markdown("---")
    
    # Trend analysis
    st.markdown("#### 📈 Trend Analysis")
    
    trends = generator.identify_trends()
    
    if trends:
        trends_df = pd.DataFrame(trends)
        st.dataframe(trends_df, use_container_width=True)
    else:
        st.info("No trends identified")
    
    st.markdown("---")
    
    # Outlier summary
    st.markdown("#### 🎯 Outlier Detection")
    
    outliers = generator.identify_outliers_summary()
    
    if outliers:
        outliers_df = pd.DataFrame(outliers)
        st.dataframe(outliers_df, use_container_width=True)
    else:
        st.info("No significant outliers detected")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("#### 📋 Recommendations")
    
    recommendations = generator.generate_recommendations()
    
    for idx, rec in enumerate(recommendations, 1):
        st.write(f"{idx}. {rec}")
    
    st.markdown("---")
    
    # Correlation insights
    st.markdown("#### 🔗 Correlation Insights")
    
    corr_insights = generator.get_correlation_insights(threshold=0.7)
    
    if corr_insights:
        for idx, insight in enumerate(corr_insights, 1):
            st.write(f"{idx}. {insight}")
    else:
        st.info("No significant correlations found")


def display_tab_reports():
    """Display report generation tab"""
    st.markdown("### 📄 Report Generation")
    
    df = st.session_state.dataframe
    analyzer = StatisticalAnalysis(df)
    generator = InsightGenerator(df)
    
    st.markdown("#### 📊 Generate Reports")
    
    col1, col2, col3 = st.columns(3)
    
    # Excel Report
    with col1:
        st.markdown("##### 📋 Excel Report")
        
        if st.button("Generate Excel", use_container_width=True):
            with st.spinner("Generating Excel report..."):
                report_gen = ReportGenerator(df, "Data Analysis Report")
                
                stats_summary = analyzer.get_summary_statistics().to_dict()
                recommendations = generator.generate_recommendations()
                
                excel_data = {
                    'Summary Stats': analyzer.get_summary_statistics(),
                    'Correlation': analyzer.get_correlation_matrix(),
                }
                
                excel_bytes = report_gen.generate_excel_report(
                    "analytics_report.xlsx",
                    excel_data
                )
                
                if excel_bytes:
                    st.download_button(
                        label="⬇️ Download Excel Report",
                        data=excel_bytes,
                        file_name="analytics_report.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                    st.success("✅ Report generated!")
    
    # CSV Export
    with col2:
        st.markdown("##### 📊 CSV Export")
        
        if st.button("Generate CSV", use_container_width=True):
            with st.spinner("Generating CSV..."):
                report_gen = ReportGenerator(df)
                csv_bytes = report_gen.generate_csv_export()
                
                if csv_bytes:
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_bytes,
                        file_name="analytics_data.csv",
                        mime="text/csv"
                    )
                    st.success("✅ CSV generated!")
    
    # Text Report
    with col3:
        st.markdown("##### 📝 Text Report")
        
        if st.button("Generate Text Report", use_container_width=True):
            with st.spinner("Generating text report..."):
                report_gen = ReportGenerator(df)
                recommendations = generator.generate_recommendations()
                
                text_report = report_gen.create_summary_report(
                    {},
                    recommendations
                )
                
                st.download_button(
                    label="⬇️ Download Text Report",
                    data=text_report,
                    file_name="analytics_report.txt",
                    mime="text/plain"
                )
                st.success("✅ Report generated!")
    
    st.markdown("---")
    
    # Report preview
    st.markdown("#### 👁️ Report Preview")
    
    if st.checkbox("Show Text Report Preview"):
        report_gen = ReportGenerator(df)
        recommendations = generator.generate_recommendations()
        
        text_report = report_gen.create_summary_report({}, recommendations)
        
        st.text_area(
            "Report Content",
            text_report,
            height=400,
            disabled=True
        )


def main():
    """Main application function"""
    initialize_session_state()
    
    if not st.session_state.logged_in:
        display_login_page()
    else:
        display_dashboard()


if __name__ == "__main__":
    main()
