"""
Utilities Module
Common utility functions and helpers
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import time
from typing import Optional, List


def load_sample_data(sample_type: str = 'sales') -> pd.DataFrame:
    """
    Load sample dataset for demonstration
    
    Args:
        sample_type (str): Type of sample data ('sales', 'employees', 'general')
        
    Returns:
        pd.DataFrame: Sample dataframe
    """
    if sample_type == 'sales':
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        data = {
            'Date': dates,
            'Product': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Sales': np.random.randint(1000, 10000, 100),
            'Quantity': np.random.randint(1, 50, 100),
            'Profit': np.random.randint(100, 2000, 100),
            'Customer_Satisfaction': np.random.uniform(3.0, 5.0, 100)
        }
        return pd.DataFrame(data)
    
    elif sample_type == 'employees':
        np.random.seed(42)
        data = {
            'Employee_ID': range(1, 51),
            'Name': [f'Employee_{i}' for i in range(1, 51)],
            'Department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 50),
            'Salary': np.random.randint(30000, 100000, 50),
            'Experience_Years': np.random.randint(1, 20, 50),
            'Performance_Rating': np.random.uniform(2.0, 5.0, 50),
            'Joining_Date': pd.date_range('2015-01-01', periods=50, freq='M')
        }
        return pd.DataFrame(data)
    
    else:  # general
        np.random.seed(42)
        data = {
            'ID': range(1, 101),
            'Category': np.random.choice(['A', 'B', 'C'], 100),
            'Value1': np.random.randint(0, 100, 100),
            'Value2': np.random.uniform(0, 50, 100),
            'Value3': np.random.randint(10, 1000, 100),
            'Status': np.random.choice(['Active', 'Inactive'], 100)
        }
        return pd.DataFrame(data)


def get_numeric_columns(dataframe: pd.DataFrame) -> List[str]:
    """
    Get list of numeric columns
    
    Args:
        dataframe (pd.DataFrame): Input dataframe
        
    Returns:
        List[str]: List of numeric column names
    """
    return dataframe.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(dataframe: pd.DataFrame) -> List[str]:
    """
    Get list of categorical columns
    
    Args:
        dataframe (pd.DataFrame): Input dataframe
        
    Returns:
        List[str]: List of categorical column names
    """
    return dataframe.select_dtypes(include=['object', 'category']).columns.tolist()


def bold_text(text: str) -> str:
    """Format text as bold for Streamlit"""
    return f"**{text}**"


def highlight_positive(value: float) -> str:
    """Highlight positive values in green, negative in red"""
    if value > 0:
        return f"🟢 {value:.2f}"
    elif value < 0:
        return f"🔴 {value:.2f}"
    else:
        return f"⚪ {value:.2f}"


def display_loading_animation(message: str = "Processing...", duration: int = 3):
    """
    Display loading animation with spinner
    
    Args:
        message (str): Loading message
        duration (int): Duration in seconds
    """
    with st.spinner(f"{message}"):
        time.sleep(duration)


def format_number(number: float, decimals: int = 2) -> str:
    """
    Format number with proper spacing and decimals
    
    Args:
        number (float): Number to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    return f"{number:,.{decimals}f}"


def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        current (float): Current value
        previous (float): Previous value
        
    Returns:
        float: Percentage change
    """
    if previous == 0:
        return 0
    return ((current - previous) / abs(previous)) * 100


def create_dataframe_summary(dataframe: pd.DataFrame) -> dict:
    """
    Create comprehensive summary of dataframe
    
    Args:
        dataframe (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Summary information
    """
    return {
        'rows': len(dataframe),
        'columns': len(dataframe.columns),
        'numeric_cols': len(dataframe.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(dataframe.select_dtypes(include=['object', 'category']).columns),
        'memory_mb': dataframe.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': dataframe.isnull().sum().sum(),
        'duplicates': dataframe.duplicated().sum()
    }


def validate_dataframe(dataframe: pd.DataFrame) -> tuple:
    """
    Validate dataframe for analysis
    
    Args:
        dataframe (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if dataframe is None or len(dataframe) == 0:
        return False, "Dataset is empty"
    
    if len(dataframe.columns) == 0:
        return False, "Dataset has no columns"
    
    if dataframe.isnull().sum().sum() == len(dataframe) * len(dataframe.columns):
        return False, "Dataset contains only missing values"
    
    return True, "Dataset is valid"


def get_file_download_link(file_bytes: bytes, filename: str, file_type: str = "xlsx") -> str:
    """
    Create download link for file
    
    Args:
        file_bytes (bytes): File content as bytes
        filename (str): Filename
        file_type (str): File type (xlsx, csv, pdf)
        
    Returns:
        str: HTML download link
    """
    if file_type == "xlsx":
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_type == "csv":
        mime_type = "text/csv"
    elif file_type == "pdf":
        mime_type = "application/pdf"
    else:
        mime_type = "application/octet-stream"
    
    return mime_type


def get_time_based_greeting() -> str:
    """
    Get greeting based on current time
    
    Returns:
        str: Time-appropriate greeting
    """
    current_hour = datetime.now().hour
    
    if current_hour < 12:
        return "Good Morning! 🌅"
    elif current_hour < 18:
        return "Good Afternoon! ☀️"
    else:
        return "Good Evening! 🌙"
