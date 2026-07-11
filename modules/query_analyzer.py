"""
Query Analyzer Module
Natural Language Query Analysis for Dataset
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import json


class QueryAnalyzer:
    """
    Analyzes natural language queries on datasets
    Provides answers and suggests follow-up questions
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize QueryAnalyzer with dataframe
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
        """
        self.df = dataframe
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.all_cols = self.df.columns.tolist()
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze and execute a natural language query
        
        Args:
            query (str): User's natural language query
            
        Returns:
            dict: Analysis results with answer, data, and chart suggestion
        """
        query_lower = query.lower().strip()
        
        # Aggregation queries (sum, avg, max, min)
        if any(keyword in query_lower for keyword in ['sum', 'total']):
            return self._handle_sum_query(query_lower)
        
        elif any(keyword in query_lower for keyword in ['average', 'avg', 'mean']):
            return self._handle_average_query(query_lower)
        
        elif any(keyword in query_lower for keyword in ['maximum', 'max', 'highest', 'largest', 'best']):
            return self._handle_max_query(query_lower)
        
        elif any(keyword in query_lower for keyword in ['minimum', 'min', 'lowest', 'smallest']):
            return self._handle_min_query(query_lower)
        
        elif any(keyword in query_lower for keyword in ['how many', 'count', 'total rows', 'records']):
            return self._handle_count_query(query_lower)
        
        elif any(keyword in query_lower for keyword in ['filter', 'where', 'show me', 'find']):
            return self._handle_filter_query(query_lower)
        
        elif any(keyword in query_lower for keyword in ['trend', 'growth', 'increase', 'decrease', 'change']):
            return self._handle_trend_query(query_lower)
        
        elif any(keyword in query_lower for keyword in ['distribution', 'compare', 'difference']):
            return self._handle_distribution_query(query_lower)
        
        elif any(keyword in query_lower for keyword in ['summarize', 'summary', 'overview', 'tell me about']):
            return self._handle_summary_query(query_lower)
        
        elif any(keyword in query_lower for keyword in ['correlation', 'related', 'relationship']):
            return self._handle_correlation_query(query_lower)
        
        else:
            return self._handle_generic_query(query_lower)
    
    def _handle_sum_query(self, query: str) -> Dict:
        """Handle sum/total queries"""
        results = []
        
        for col in self.numeric_cols:
            if col.lower() in query or query.count(col.lower()) > 0:
                total = self.df[col].sum()
                results.append({
                    'column': col,
                    'value': total,
                    'formatted': f"${total:,.2f}" if 'price' in col.lower() or 'sales' in col.lower() else f"{total:,.0f}"
                })
        
        if not results:
            # Default to all numeric columns
            results = [{'column': col, 'value': self.df[col].sum(), 'formatted': f"{self.df[col].sum():,.0f}"} 
                      for col in self.numeric_cols[:3]]
        
        answer = f"Here are the totals:\n"
        for r in results:
            answer += f"• {r['column']}: {r['formatted']}\n"
        
        return {
            'answer': answer,
            'data': results,
            'chart_type': 'bar',
            'table_data': pd.DataFrame(results),
            'success': True
        }
    
    def _handle_average_query(self, query: str) -> Dict:
        """Handle average queries"""
        results = []
        
        for col in self.numeric_cols:
            if col.lower() in query or len(self.numeric_cols) <= 3:
                avg = self.df[col].mean()
                results.append({
                    'column': col,
                    'value': avg,
                    'formatted': f"${avg:,.2f}" if 'price' in col.lower() else f"{avg:,.2f}"
                })
        
        if not results:
            results = [{'column': col, 'value': self.df[col].mean(), 'formatted': f"{self.df[col].mean():,.2f}"} 
                      for col in self.numeric_cols[:3]]
        
        answer = f"Average values:\n"
        for r in results:
            answer += f"• {r['column']}: {r['formatted']}\n"
        
        return {
            'answer': answer,
            'data': results,
            'table_data': pd.DataFrame(results),
            'success': True
        }
    
    def _handle_max_query(self, query: str) -> Dict:
        """Handle maximum/highest queries"""
        results = []
        
        for col in self.numeric_cols:
            max_val = self.df[col].max()
            max_idx = self.df[col].idxmax()
            results.append({
                'column': col,
                'max_value': max_val,
                'row_index': max_idx,
                'formatted': f"${max_val:,.2f}" if 'price' in col.lower() else f"{max_val:,.0f}"
            })
        
        if len(results) == 0:
            return {'answer': "No numeric columns found", 'success': False}
        
        best = results[0]
        answer = f"The highest value is in column '{best['column']}': {best['formatted']}"
        
        if 'which' in query and 'row' in query:
            answer += f"\n(Row index: {best['row_index']})"
        
        return {
            'answer': answer,
            'data': results,
            'table_data': pd.DataFrame(results),
            'success': True
        }
    
    def _handle_min_query(self, query: str) -> Dict:
        """Handle minimum/lowest queries"""
        results = []
        
        for col in self.numeric_cols:
            min_val = self.df[col].min()
            min_idx = self.df[col].idxmin()
            results.append({
                'column': col,
                'min_value': min_val,
                'row_index': min_idx,
                'formatted': f"${min_val:,.2f}" if 'price' in col.lower() else f"{min_val:,.0f}"
            })
        
        if len(results) == 0:
            return {'answer': "No numeric columns found", 'success': False}
        
        lowest = results[0]
        answer = f"The lowest value is in column '{lowest['column']}': {lowest['formatted']}"
        
        return {
            'answer': answer,
            'data': results,
            'table_data': pd.DataFrame(results),
            'success': True
        }
    
    def _handle_count_query(self, query: str) -> Dict:
        """Handle count queries"""
        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        missing = self.df.isnull().sum().sum()
        duplicates = self.df.duplicated().sum()
        
        answer = f"""Dataset Summary:
• Total Rows: {total_rows:,}
• Total Columns: {total_cols}
• Missing Values: {missing:,}
• Duplicate Rows: {duplicates:,}"""
        
        return {
            'answer': answer,
            'data': {
                'rows': total_rows,
                'columns': total_cols,
                'missing': missing,
                'duplicates': duplicates
            },
            'success': True
        }
    
    def _handle_filter_query(self, query: str) -> Dict:
        """Handle filter queries"""
        try:
            # Simple filter parsing
            result_df = self.df.copy()
            
            # Try to match column names and values
            patterns = [
                (r'(\w+)\s*(?:is|=|equal|equals)\s*([^\s]+)', 'equals'),
                (r'(\w+)\s*(?:>|greater)\s*(\d+)', 'greater'),
                (r'(\w+)\s*(?:<|less)\s*(\d+)', 'less'),
            ]
            
            for pattern, op_type in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    col, val = matches[0]
                    col = col.lower()
                    
                    # Find matching column
                    matching_col = None
                    for c in self.df.columns:
                        if c.lower() == col or col in c.lower():
                            matching_col = c
                            break
                    
                    if matching_col:
                        if op_type == 'equals':
                            result_df = result_df[result_df[matching_col].astype(str) == val]
                        elif op_type == 'greater':
                            result_df = result_df[result_df[matching_col] > float(val)]
                        elif op_type == 'less':
                            result_df = result_df[result_df[matching_col] < float(val)]
            
            if len(result_df) == 0:
                answer = "No rows match your filter criteria."
                return {'answer': answer, 'data': result_df, 'success': False}
            
            answer = f"Found {len(result_df):,} rows matching your criteria:\n"
            return {
                'answer': answer,
                'data': result_df,
                'table_data': result_df.head(10),
                'success': True
            }
        except Exception as e:
            return {'answer': f"Could not parse filter: {str(e)}", 'success': False}
    
    def _handle_trend_query(self, query: str) -> Dict:
        """Handle trend queries"""
        trends = []
        
        for col in self.numeric_cols:
            if len(self.df) > 1:
                first_half = self.df[col].iloc[:len(self.df)//2].mean()
                second_half = self.df[col].iloc[len(self.df)//2:].mean()
                
                if first_half > 0:
                    change = ((second_half - first_half) / first_half) * 100
                else:
                    change = 0
                
                trend_dir = "📈 Increasing" if change > 0 else "📉 Decreasing" if change < 0 else "➡️ Stable"
                
                trends.append({
                    'column': col,
                    'change_percent': change,
                    'trend': trend_dir,
                    'first_half': first_half,
                    'second_half': second_half
                })
        
        if not trends:
            return {'answer': "No trends found", 'success': False}
        
        answer = "Trend Analysis:\n"
        for t in trends[:5]:
            answer += f"• {t['column']}: {t['trend']} ({t['change_percent']:.1f}%)\n"
        
        return {
            'answer': answer,
            'data': trends,
            'table_data': pd.DataFrame(trends),
            'chart_type': 'bar',
            'success': True
        }
    
    def _handle_distribution_query(self, query: str) -> Dict:
        """Handle distribution queries"""
        answer = "Distribution Analysis:\n"
        
        for col in self.numeric_cols[:5]:
            median = self.df[col].median()
            mean = self.df[col].mean()
            std = self.df[col].std()
            skew = self.df[col].skew()
            
            answer += f"\n{col}:\n"
            answer += f"  • Mean: {mean:.2f}\n"
            answer += f"  • Median: {median:.2f}\n"
            answer += f"  • Std Dev: {std:.2f}\n"
            answer += f"  • Skewness: {skew:.2f}\n"
        
        return {
            'answer': answer,
            'success': True
        }
    
    def _handle_summary_query(self, query: str) -> Dict:
        """Handle summary queries"""
        answer = f"""📊 Dataset Overview:
        
Dimensions: {len(self.df):,} rows × {len(self.df.columns)} columns

Columns: {', '.join(self.df.columns.tolist()[:5])}{'...' if len(self.df.columns) > 5 else ''}

Numeric Columns: {len(self.numeric_cols)}
Categorical Columns: {len(self.categorical_cols)}

Missing Values: {self.df.isnull().sum().sum():,}
Duplicate Rows: {self.df.duplicated().sum():,}"""
        
        return {
            'answer': answer,
            'success': True
        }
    
    def _handle_correlation_query(self, query: str) -> Dict:
        """Handle correlation queries"""
        if len(self.numeric_cols) < 2:
            return {'answer': "Need at least 2 numeric columns for correlation", 'success': False}
        
        corr = self.df[self.numeric_cols].corr().unstack().reset_index()
        corr.columns = ['Column1', 'Column2', 'Correlation']
        corr = corr[corr['Column1'] != corr['Column2']]
        corr['Correlation'] = corr['Correlation'].abs()
        corr = corr.sort_values('Correlation', ascending=False).head(5)
        
        answer = "Top Correlations:\n"
        for _, row in corr.iterrows():
            answer += f"• {row['Column1']} ↔ {row['Column2']}: {row['Correlation']:.3f}\n"
        
        return {
            'answer': answer,
            'data': corr,
            'table_data': corr,
            'success': True
        }
    
    def _handle_generic_query(self, query: str) -> Dict:
        """Handle generic queries"""
        answer = f"""I understand you're asking about your data. Here are some things I can help with:

🔢 **Data Topics:**
• "What is the sum of [column]?"
• "Show me the average [column]"
• "Which row has the highest [column]?"
• "What's the trend in [column]?"
• "Filter rows where [column] > 100"

📊 **Analysis:**
• "Summarize this dataset"
• "Show me trends"
• "What are the correlations?"
• "How many rows have missing values?"

Try asking a specific question about your data!"""
        
        return {
            'answer': answer,
            'success': False
        }
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions based on dataset"""
        suggestions = []
        
        if len(self.numeric_cols) > 0:
            col = self.numeric_cols[0]
            suggestions.append(f"What is the average {col}?")
            suggestions.append(f"Which row has the highest {col}?")
        
        if len(self.categorical_cols) > 0:
            col = self.categorical_cols[0]
            if len(self.df[col].unique()) <= 10:
                suggestions.append(f"Show distribution of {col}")
        
        suggestions.extend([
            "Summarize this dataset",
            "What trends do you see?",
            "How many rows have missing values?",
            "Show me the top correlations"
        ])
        
        return suggestions[:5]
