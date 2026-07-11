"""
Insight Generator Module
Generates AI-powered business insights and recommendations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class InsightGenerator:
    """
    Class to generate automated insights from data
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize InsightGenerator with dataframe
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
        """
        self.df = dataframe
    
    def generate_top_columns(self, metric: str = 'variance', top_n: int = 5) -> List[Dict]:
        """
        Identify top performing columns based on variance or other metrics
        
        Args:
            metric (str): 'variance', 'mean', 'std'
            top_n (int): Number of top columns to return
            
        Returns:
            List[Dict]: Top columns with scores
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return []
        
        scores = {}
        
        if metric == 'variance':
            scores = numeric_df.var()
        elif metric == 'mean':
            scores = numeric_df.mean()
        elif metric == 'std':
            scores = numeric_df.std()
        else:
            scores = numeric_df.var()
        
        top_cols = scores.nlargest(top_n)
        
        insights = [
            {
                'column': col,
                'score': float(score),
                'metric': metric
            }
            for col, score in top_cols.items()
        ]
        
        return insights
    
    def identify_trends(self, numeric_columns: List[str] = None) -> List[Dict]:
        """
        Identify growth and decline trends in columns
        
        Args:
            numeric_columns (List[str]): Columns to analyze (None = all numeric)
            
        Returns:
            List[Dict]: Trend analysis for each column
        """
        if numeric_columns is None:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        trends = []
        
        for col in numeric_columns:
            if col in self.df.columns:
                values = self.df[col].dropna().values
                
                if len(values) > 1:
                    first_half = values[:len(values)//2].mean()
                    second_half = values[len(values)//2:].mean()
                    
                    if first_half > 0:
                        growth_rate = ((second_half - first_half) / first_half) * 100
                    else:
                        growth_rate = 0
                    
                    trend_direction = 'Increasing' if growth_rate > 0 else 'Decreasing'
                    trend_strength = abs(growth_rate)
                    
                    trends.append({
                        'column': col,
                        'trend': trend_direction,
                        'growth_rate': round(growth_rate, 2),
                        'trend_strength': round(trend_strength, 2),
                        'first_half_avg': round(first_half, 4),
                        'second_half_avg': round(second_half, 4)
                    })
        
        return sorted(trends, key=lambda x: abs(x['growth_rate']), reverse=True)
    
    def identify_outliers_summary(self, numeric_columns: List[str] = None) -> List[Dict]:
        """
        Identify and summarize outliers
        
        Args:
            numeric_columns (List[str]): Columns to analyze
            
        Returns:
            List[Dict]: Outlier summary for each column
        """
        if numeric_columns is None:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_summary = []
        
        for col in numeric_columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[
                    (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                ]
                
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(self.df)) * 100 if len(self.df) > 0 else 0
                
                if outlier_count > 0:
                    outliers_summary.append({
                        'column': col,
                        'outlier_count': outlier_count,
                        'outlier_percentage': round(outlier_percentage, 2),
                        'lower_bound': round(lower_bound, 4),
                        'upper_bound': round(upper_bound, 4),
                        'min_outlier': round(outliers[col].min(), 4),
                        'max_outlier': round(outliers[col].max(), 4)
                    })
        
        return sorted(outliers_summary, key=lambda x: x['outlier_count'], reverse=True)
    
    def analyze_missing_data(self) -> List[Dict]:
        """
        Analyze missing data patterns for each column
        
        Returns:
            List[Dict]: Missing data analysis for each column with missing values
        """
        missing_analysis = []
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            
            if missing_count > 0:
                missing_percentage = (missing_count / len(self.df)) * 100
                
                missing_analysis.append({
                    'column': col,
                    'missing': missing_count,
                    'percentage': round(missing_percentage, 2),
                    'total_rows': len(self.df)
                })
        
        # Sort by missing percentage (highest first)
        return sorted(missing_analysis, key=lambda x: x['percentage'], reverse=True)
    
    def generate_recommendations(self) -> List[str]:
        """
        Generate personalized recommendations based on data characteristics
        
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        # Check for missing values
        missing_pct = (self.df.isnull().sum().sum() / 
                      (self.df.shape[0] * self.df.shape[1])) * 100
        if missing_pct > 0:
            recommendations.append(
                f"⚠️ Dataset contains {missing_pct:.2f}% missing values. "
                "Consider using imputation techniques."
            )
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            recommendations.append(
                f"🔄 Found {duplicates} duplicate rows. "
                "Consider removing them for cleaner analysis."
            )
        
        # Check for outliers
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[
                (self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)
            ]
            total_outliers += len(outliers)
        
        if total_outliers > 0:
            recommendations.append(
                f"📊 Detected {total_outliers} potential outliers. "
                "Investigate or remove them for better model performance."
            )
        
        # Check for column correlation
        correlation_matrix = self.df.select_dtypes(include=[np.number]).corr()
        high_correlations = 0
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.9:
                    high_correlations += 1
        
        if high_correlations > 0:
            recommendations.append(
                f"🔗 Found {high_correlations} highly correlated column pairs. "
                "Consider removing redundant features."
            )
        
        # Check data distribution
        skew_values = self.df.select_dtypes(include=[np.number]).skew()
        highly_skewed = (abs(skew_values) > 2).sum()
        
        if highly_skewed > 0:
            recommendations.append(
                f"📈 {highly_skewed} columns are highly skewed. "
                "Consider log transformation or scaling."
            )
        
        # Check for categorical imbalance
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            if len(value_counts) > 1:
                max_pct = (value_counts.iloc[0] / len(self.df)) * 100
                if max_pct > 80:
                    recommendations.append(
                        f"⚠️ Column '{col}' is imbalanced (80% same value). "
                        "This may affect classification models."
                    )
        
        if not recommendations:
            recommendations.append(
                "✅ Dataset quality looks good! Ready for advanced analysis."
            )
        
        return recommendations
    
    def generate_correlation_insights(self, threshold: float = 0.7) -> List[str]:
        """
        Extract meaningful insights from correlation analysis
        
        Args:
            threshold (float): Correlation threshold for significance (default: 0.7)
            
        Returns:
            List[str]: Correlation insights as human-readable text
        """
        try:
            # Get only numeric columns for correlation analysis
            numeric_df = self.df.select_dtypes(include=[np.number])
            
            # Need at least 2 numeric columns for meaningful correlations
            if len(numeric_df.columns) < 2:
                return ["📊 Need at least 2 numeric columns for correlation analysis"]
            
            # Calculate correlation matrix
            correlation = numeric_df.corr()
            
            insights = []
            processed_pairs = set()  # Track processed pairs to avoid duplicates
            
            # Analyze each unique pair of columns
            for i in range(len(correlation.columns)):
                for j in range(i+1, len(correlation.columns)):
                    corr_value = correlation.iloc[i, j]
                    
                    # Check for strong correlations (positive or negative)
                    if abs(corr_value) > threshold:
                        col1 = correlation.columns[i]
                        col2 = correlation.columns[j]
                        
                        # Create unique pair identifier to avoid duplicates
                        pair_key = tuple(sorted([col1, col2]))
                        if pair_key in processed_pairs:
                            continue
                        processed_pairs.add(pair_key)
                        
                        # Generate human-readable insight
                        if corr_value > 0:
                            insight = f"📈 '{col1}' and '{col2}' are strongly positively correlated ({corr_value:.3f})"
                        else:
                            insight = f"📉 '{col1}' and '{col2}' are strongly negatively correlated ({corr_value:.3f})"
                        
                        insights.append(insight)
            
            # Return insights or default message
            if not insights:
                insights.append("📊 No strong correlations found between numeric columns")
            
            return insights
            
        except Exception as e:
            # Safe error handling
            return [f"⚠️ Error analyzing correlations: {str(e)}"]
    
    def generate_business_insights(self) -> List[str]:
        """
        Generate actionable business insights based on data analysis
        
        Analyzes numeric columns to find top performing metrics, detect low performance,
        check missing values, and generate business recommendations.
        
        Returns:
            List[str]: List of business insights and recommendations
        """
        insights = []
        
        try:
            # Get numeric columns for analysis
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return ["📊 No numeric columns found for business analysis"]
            
            # 1. Analyze top performing metrics (highest means)
            if len(numeric_cols) > 0:
                means = self.df[numeric_cols].mean()
                top_performers = means.nlargest(3)
                
                for col, value in top_performers.items():
                    if value > 0:
                        insights.append(f"📈 Top performer: '{col}' averages {value:.2f} - excellent performance")
            
            # 2. Detect low performance metrics (lowest means, excluding zeros/negatives)
            positive_means = means[means > 0]
            if len(positive_means) > 0:
                low_performers = positive_means.nsmallest(2)
                
                for col, value in low_performers.items():
                    if value < positive_means.mean() * 0.5:  # Below 50% of average
                        insights.append(f"⚠️ Low performer: '{col}' averages only {value:.2f} - needs attention")
            
            # 3. Check for high variance (volatility)
            variances = self.df[numeric_cols].var()
            high_variance = variances[variances > variances.quantile(0.75)]
            
            for col, var_value in high_variance.items():
                std_dev = np.sqrt(var_value)
                mean_val = self.df[col].mean()
                cv = std_dev / mean_val if mean_val != 0 else 0  # Coefficient of variation
                
                if cv > 0.5:  # High variability
                    insights.append(f"📊 High volatility: '{col}' shows significant variation (CV: {cv:.2f}) - consider stability measures")
            
            # 4. Analyze missing values impact
            missing_analysis = self.analyze_missing_data()
            if missing_analysis:
                # Focus on columns with high missing percentages
                high_missing = [item for item in missing_analysis if item['percentage'] > 10]
                
                for missing_info in high_missing[:2]:  # Limit to top 2
                    col = missing_info['column']
                    pct = missing_info['percentage']
                    insights.append(f"🚨 Data quality issue: '{col}' has {pct:.1f}% missing values - impacts analysis reliability")
            
            # 5. Generate business recommendations based on data patterns
            total_rows = len(self.df)
            
            # Check for data completeness
            completeness = (1 - self.df.isnull().sum().sum() / (total_rows * len(self.df.columns))) * 100
            if completeness < 80:
                insights.append(f"📋 Data completeness: Only {completeness:.1f}% complete - recommend data cleaning initiatives")
            
            # Check for potential growth opportunities
            if len(numeric_cols) >= 2:
                # Look for columns that might indicate growth potential
                growth_indicators = []
                for col in numeric_cols:
                    if 'revenue' in col.lower() or 'sales' in col.lower() or 'profit' in col.lower():
                        growth_indicators.append(col)
                
                if growth_indicators:
                    insights.append(f"💰 Growth focus: Monitor {', '.join(growth_indicators)} for business performance tracking")
            
            # Check for operational efficiency indicators
            efficiency_indicators = []
            for col in numeric_cols:
                if any(keyword in col.lower() for keyword in ['cost', 'expense', 'efficiency', 'productivity']):
                    efficiency_indicators.append(col)
            
            if efficiency_indicators:
                insights.append(f"⚡ Efficiency focus: Track {', '.join(efficiency_indicators)} for operational improvements")
            
            # Default insights if none generated
            if not insights:
                insights.append("✅ Business metrics look stable - continue monitoring key performance indicators")
                insights.append("📊 Consider setting up automated alerts for significant metric changes")
            
            return insights[:10]  # Limit to top 10 insights
            
        except Exception as e:
            return [f"⚠️ Error generating business insights: {str(e)}"]
    
    def summarize_categories(self, categorical_column: str) -> Dict:
        """
        Summarize categorical distribution
        
        Args:
            categorical_column (str): Column to summarize
            
        Returns:
            dict: Category summary
        """
        if categorical_column not in self.df.columns:
            return {}
        
        value_counts = self.df[categorical_column].value_counts()
        
        summary = {
            'total_categories': len(value_counts),
            'top_category': value_counts.index[0],
            'top_category_count': value_counts.iloc[0],
            'distribution': value_counts.to_dict()
        }
        
        return summary
