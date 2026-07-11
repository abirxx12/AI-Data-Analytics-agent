"""
Machine Learning Analysis Module
Performs ML predictions and trend analysis
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Optional, List


class MLAnalyzer:
    """
    Class to perform ML analysis and predictions
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize MLAnalyzer with dataframe
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
        """
        self.df = dataframe
        self.models = {}
        self.predictions = {}
    
    def perform_linear_regression(self, x_col: str, y_col: str) -> Dict:
        """
        Perform linear regression analysis
        
        Args:
            x_col (str): Feature column
            y_col (str): Target column
            
        Returns:
            dict: Model results and metrics
        """
        try:
            X = self.df[[x_col]].dropna()
            y = self.df[y_col].loc[X.index].dropna()
            X = X.loc[y.index]
            
            if len(X) < 2:
                return {'error': 'Insufficient data'}
            
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            results = {
                'model': model,
                'coefficient': model.coef_[0],
                'intercept': model.intercept_,
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'predictions': y_pred,
                'actual': y.values,
                'x_values': X[x_col].values
            }
            
            self.models['linear_regression'] = results
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def perform_polynomial_regression(self, x_col: str, y_col: str, degree: int = 2) -> Dict:
        """
        Perform polynomial regression analysis
        
        Args:
            x_col (str): Feature column
            y_col (str): Target column
            degree (int): Polynomial degree
            
        Returns:
            dict: Model results and metrics
        """
        try:
            X = self.df[[x_col]].dropna()
            y = self.df[y_col].loc[X.index].dropna()
            X = X.loc[y.index]
            
            if len(X) < degree + 2:
                return {'error': 'Insufficient data for polynomial degree'}
            
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            y_pred = model.predict(X_poly)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            results = {
                'model': model,
                'polynomial_transformer': poly,
                'degree': degree,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'predictions': y_pred,
                'actual': y.values,
                'x_values': X[x_col].values
            }
            
            self.models['polynomial_regression'] = results
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def perform_random_forest_regression(self, feature_cols: List[str], 
                                         target_col: str, n_estimators: int = 100) -> Dict:
        """
        Perform random forest regression
        
        Args:
            feature_cols (List[str]): Feature columns
            target_col (str): Target column
            n_estimators (int): Number of trees
            
        Returns:
            dict: Model results and metrics
        """
        try:
            # Prepare data
            df_clean = self.df[feature_cols + [target_col]].dropna()
            
            if len(df_clean) < 2:
                return {'error': 'Insufficient data'}
            
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            
            # Handle categorical features
            categorical_cols = X.select_dtypes(include=['object']).columns
            X_processed = X.copy()
            
            for col in categorical_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            
            # Train model
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_processed, y)
            
            y_pred = model.predict(X_processed)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results = {
                'model': model,
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'actual': y.values
            }
            
            self.models['random_forest_regression'] = results
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def perform_random_forest_classification(self, feature_cols: List[str], 
                                            target_col: str, n_estimators: int = 100) -> Dict:
        """
        Perform random forest classification
        
        Args:
            feature_cols (List[str]): Feature columns
            target_col (str): Target column
            n_estimators (int): Number of trees
            
        Returns:
            dict: Model results and metrics
        """
        try:
            df_clean = self.df[feature_cols + [target_col]].dropna()
            
            if len(df_clean) < 2:
                return {'error': 'Insufficient data'}
            
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            
            # Handle categorical features
            categorical_cols = X.select_dtypes(include=['object']).columns
            X_processed = X.copy()
            
            for col in categorical_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            
            # Encode target if categorical
            if pd.api.types.is_object_dtype(y):
                le_y = LabelEncoder()
                y_encoded = le_y.fit_transform(y)
            else:
                y_encoded = y
            
            # Train model
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_processed, y_encoded)
            
            y_pred = model.predict(X_processed)
            
            accuracy = accuracy_score(y_encoded, y_pred)
            precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'actual': y_encoded
            }
            
            self.models['random_forest_classification'] = results
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def predict_trends(self, column: str, periods: int = 5) -> Dict:
        """
        Predict trends in a time-series column
        
        Args:
            column (str): Column to predict
            periods (int): Number of periods to predict
            
        Returns:
            dict: Trend analysis and predictions
        """
        try:
            data = self.df[column].dropna().values
            
            if len(data) < 3:
                return {'error': 'Insufficient data'}
            
            # Create index array
            X = np.arange(len(data)).reshape(-1, 1)
            y = data
            
            # Fit linear model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict future values
            future_indices = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            future_predictions = model.predict(future_indices)
            
            # Calculate trend
            trend = 'Increasing' if model.coef_[0] > 0 else 'Decreasing'
            trend_strength = abs(model.coef_[0])
            
            # Calculate growth rate
            growth_rate = ((data[-1] - data[0]) / data[0] * 100) if data[0] != 0 else 0
            
            results = {
                'current_values': data,
                'predictions': future_predictions,
                'trend': trend,
                'trend_strength': trend_strength,
                'growth_rate': growth_rate,
                'slope': model.coef_[0],
                'intercept': model.intercept_
            }
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained models
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        for model_name, results in self.models.items():
            if 'feature_importance' in results:
                return results['feature_importance']
        
        return pd.DataFrame()
