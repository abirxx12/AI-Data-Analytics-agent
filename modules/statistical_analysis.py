"""
Statistical Analysis Module
Computes statistical measures and generates insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class StatisticalAnalysis:
    """
    Class to perform statistical analysis on datasets
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize StatisticalAnalysis with dataframe
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
        """
        self.df = dataframe
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for numeric columns
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        summary = pd.DataFrame({
            'Count': numeric_df.count(),
            'Mean': numeric_df.mean(),
            'Median': numeric_df.median(),
            'Mode': numeric_df.mode().iloc[0] if len(numeric_df.mode()) > 0 else np.nan,
            'Std Dev': numeric_df.std(),
            'Min': numeric_df.min(),
            'Max': numeric_df.max(),
            'Q1': numeric_df.quantile(0.25),
            'Q3': numeric_df.quantile(0.75),
            'Skewness': numeric_df.skew(),
            'Kurtosis': numeric_df.kurtosis()
        }).round(4)
        
        return summary
    
    def get_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns
        
        Args:
            method (str): 'pearson', 'spearman', or 'kendall'
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation = numeric_df.corr(method=method)
        return correlation.round(4)
    
    def get_column_statistics(self, column: str) -> Dict:
        """
        Get detailed statistics for a specific column
        
        Args:
            column (str): Column name
            
        Returns:
            dict: Detailed statistics
        """
        if column not in self.df.columns:
            return {}
        
        col_data = self.df[column]
        
        if pd.api.types.is_numeric_dtype(col_data):
            return {
                'Column': column,
                'Data Type': str(col_data.dtype),
                'Count': col_data.count(),
                'Null Count': col_data.isnull().sum(),
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'Mode': col_data.mode().values[0] if len(col_data.mode()) > 0 else None,
                'Std Dev': col_data.std(),
                'Variance': col_data.var(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Range': col_data.max() - col_data.min(),
                'Q1': col_data.quantile(0.25),
                'Q2': col_data.quantile(0.50),
                'Q3': col_data.quantile(0.75),
                'IQR': col_data.quantile(0.75) - col_data.quantile(0.25),
                'Skewness': col_data.skew(),
                'Kurtosis': col_data.kurtosis()
            }
        else:
            value_counts = col_data.value_counts()
            return {
                'Column': column,
                'Data Type': str(col_data.dtype),
                'Count': col_data.count(),
                'Null Count': col_data.isnull().sum(),
                'Unique Values': col_data.nunique(),
                'Top Value': value_counts.index[0] if len(value_counts) > 0 else None,
                'Top Value Count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'Top 5 Values': dict(value_counts.head(5))
            }
    
    def identify_missing_pattern(self) -> Dict:
        """
        Identify patterns in missing data
        
        Returns:
            dict: Missing data patterns
        """
        missing_info = {
            'total_missing': self.df.isnull().sum().sum(),
            'total_cells': self.df.shape[0] * self.df.shape[1],
            'missing_percentage': (self.df.isnull().sum().sum() / 
                                  (self.df.shape[0] * self.df.shape[1])) * 100,
            'columns_with_missing': {}
        }
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_info['columns_with_missing'][col] = {
                    'missing_count': missing_count,
                    'missing_percentage': (missing_count / len(self.df)) * 100
                }
        
        return missing_info
    
    def find_outliers(self, column: str, method: str = 'iqr') -> List[int]:
        """
        Find outliers in a numeric column
        
        Args:
            column (str): Column name
            method (str): 'iqr' or 'zscore'
            
        Returns:
            List[int]: Indices of outlier rows
        """
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            return []
        
        col_data = self.df[column]
        outlier_indices = []
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = col_data[(col_data < lower_bound) | 
                                       (col_data > upper_bound)].index.tolist()
        
        elif method == 'zscore':
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outlier_indices = z_scores[z_scores > 3].index.tolist()
        
        return outlier_indices
    
    def get_distribution_info(self, column: str) -> Dict:
        """
        Get distribution information for a column
        
        Args:
            column (str): Column name
            
        Returns:
            dict: Distribution statistics
        """
        if column not in self.df.columns:
            return {}
        
        col_data = self.df[column]
        
        if pd.api.types.is_numeric_dtype(col_data):
            return {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis(),
                'is_normally_distributed': abs(col_data.skew()) < 2 and abs(col_data.kurtosis()) < 3
            }
        else:
            value_counts = col_data.value_counts()
            return {
                'unique_count': col_data.nunique(),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.to_dict()
            }
    
    def calculate_percentiles(self, column: str, percentiles: List[float] = None) -> Dict:
        """
        Calculate percentiles for a numeric column
        
        Args:
            column (str): Column name
            percentiles (List[float]): Percentiles to calculate (0-100)
            
        Returns:
            dict: Percentile values
        """
        if percentiles is None:
            percentiles = [10, 25, 50, 75, 90, 95, 99]
        
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            return {}
        
        col_data = self.df[column]
        percentile_values = {}
        
        for p in percentiles:
            percentile_values[f'P{int(p)}'] = col_data.quantile(p / 100)
        
        return percentile_values
