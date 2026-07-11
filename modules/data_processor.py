"""
Data Processing Module
Handles data loading, cleaning, preprocessing, and validation
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import streamlit as st


class DataProcessor:
    """
    Class to handle all data processing operations
    """
    
    def __init__(self, dataframe: pd.DataFrame = None):
        """
        Initialize DataProcessor with optional dataframe
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
        """
        self.df = dataframe
        self.original_df = dataframe.copy() if dataframe is not None else None
        self.processing_log = []
    
    def load_file(self, uploaded_file) -> bool:
        """
        Load CSV or Excel file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            bool: True if file loaded successfully
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                self.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel file.")
                return False
            
            self.original_df = self.df.copy()
            self.processing_log.append(f"File loaded: {uploaded_file.name}")
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def get_basic_info(self) -> Dict:
        """
        Get basic information about dataset
        
        Returns:
            dict: Dataset info including shape, columns, types
        """
        if self.df is None:
            return {}
        
        return {
            'rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'column_names': list(self.df.columns),
            'column_types': dict(self.df.dtypes),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
    
    def handle_missing_values(self, strategy: str = 'mean', 
                             numeric_strategy: str = 'mean',
                             categorical_strategy: str = 'mode') -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            strategy (str): Overall strategy ('mean', 'median', 'drop', 'forward_fill')
            numeric_strategy (str): Strategy for numeric columns
            categorical_strategy (str): Strategy for categorical columns
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        if self.df is None:
            return None
        
        df_processed = self.df.copy()
        missing_before = df_processed.isnull().sum().sum()
        
        if missing_before == 0:
            return df_processed
        
        # Separate numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                if numeric_strategy == 'mean':
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                elif numeric_strategy == 'median':
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                elif numeric_strategy == 'drop':
                    df_processed.dropna(subset=[col], inplace=True)
                elif numeric_strategy == 'forward_fill':
                    df_processed[col].fillna(method='ffill', inplace=True)
        
        # Handle categorical columns
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                if categorical_strategy == 'mode':
                    mode_val = df_processed[col].mode()
                    if len(mode_val) > 0:
                        df_processed[col].fillna(mode_val[0], inplace=True)
                    else:
                        df_processed[col].fillna('Unknown', inplace=True)
                elif categorical_strategy == 'drop':
                    df_processed.dropna(subset=[col], inplace=True)
        
        self.df = df_processed
        missing_after = self.df.isnull().sum().sum()
        self.processing_log.append(f"Missing values handled: {missing_before} → {missing_after}")
        
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from dataset
        
        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        if self.df is None:
            return None
        
        duplicates_before = self.df.duplicated().sum()
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        self.processing_log.append(f"Duplicates removed: {duplicates_before} rows")
        
        return self.df
    
    def remove_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from numeric columns
        
        Args:
            method (str): 'iqr' or 'zscore'
            threshold (float): Z-score threshold (used only for zscore method)
            
        Returns:
            pd.DataFrame: Dataframe without outliers
        """
        if self.df is None:
            return None
        
        df_processed = self.df.copy()
        rows_before = len(df_processed)
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_processed = df_processed[
                    (df_processed[col] >= lower_bound) & 
                    (df_processed[col] <= upper_bound)
                ]
            
            elif method == 'zscore':
                z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / 
                                 df_processed[col].std())
                df_processed = df_processed[z_scores < threshold]
        
        rows_removed = rows_before - len(df_processed)
        self.df = df_processed.reset_index(drop=True)
        self.processing_log.append(f"Outliers removed: {rows_removed} rows")
        
        return self.df
    
    def encode_categorical(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical variables to numeric
        
        Args:
            columns (List[str]): Columns to encode (None = all categorical)
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        if self.df is None:
            return None
        
        df_processed = self.df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in columns:
            if col in df_processed.columns:
                df_processed[col] = pd.factorize(df_processed[col])[0]
        
        self.df = df_processed
        self.processing_log.append(f"Categorical encoding applied to: {', '.join(columns)}")
        
        return self.df
    
    def normalize_columns(self, columns: List[str] = None, 
                         method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize numeric columns
        
        Args:
            columns (List[str]): Columns to normalize (None = all numeric)
            method (str): 'minmax' or 'standard'
            
        Returns:
            pd.DataFrame: Dataframe with normalized columns
        """
        if self.df is None:
            return None
        
        df_processed = self.df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in df_processed.columns:
                if method == 'minmax':
                    min_val = df_processed[col].min()
                    max_val = df_processed[col].max()
                    df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val)
                elif method == 'standard':
                    mean = df_processed[col].mean()
                    std = df_processed[col].std()
                    df_processed[col] = (df_processed[col] - mean) / std
        
        self.df = df_processed
        self.processing_log.append(f"Normalization applied ({method}): {', '.join(columns)}")
        
        return self.df
    
    def get_processing_log(self) -> List[str]:
        """Get log of all processing operations performed"""
        return self.processing_log
    
    def reset_to_original(self) -> pd.DataFrame:
        """Reset dataframe to original state"""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.processing_log.append("Dataset reset to original state")
        return self.df
