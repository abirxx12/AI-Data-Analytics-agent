"""
Report Generator Module
Generates PDF and Excel reports
"""

import pandas as pd
from datetime import datetime
import io
from typing import Optional


class ReportGenerator:
    """
    Class to generate reports in PDF and Excel formats
    """
    
    def __init__(self, dataframe: pd.DataFrame, title: str = "Data Analysis Report"):
        """
        Initialize ReportGenerator
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
            title (str): Report title
        """
        self.df = dataframe
        self.title = title
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_excel_report(self, filename: str, sheets_data: dict = None) -> bytes:
        """
        Generate Excel report with multiple sheets
        
        Args:
            filename (str): Output filename
            sheets_data (dict): Dictionary of sheet_name: dataframe pairs
            
        Returns:
            bytes: Excel file bytes
        """
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Add main data
                self.df.to_excel(writer, sheet_name='Data', index=False)
                
                # Add additional sheets if provided
                if sheets_data:
                    for sheet_name, data in sheets_data.items():
                        if isinstance(data, pd.DataFrame):
                            data.to_excel(writer, sheet_name=sheet_name, index=False)
                        elif isinstance(data, dict):
                            # Convert dict to DataFrame for display
                            df_data = pd.DataFrame(list(data.items()), 
                                                 columns=['Key', 'Value'])
                            df_data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Add metadata sheet
                metadata = {
                    'Report Title': [self.title],
                    'Generated': [self.timestamp],
                    'Total Rows': [len(self.df)],
                    'Total Columns': [len(self.df.columns)],
                    'File Size (MB)': [round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)]
                }
                df_metadata = pd.DataFrame(metadata)
                df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            output.seek(0)
            return output.getvalue()
        
        except Exception as e:
            return None
    
    def generate_csv_export(self) -> bytes:
        """
        Generate CSV export
        
        Returns:
            bytes: CSV file bytes
        """
        try:
            return self.df.to_csv(index=False).encode()
        except Exception as e:
            return None
    
    def create_summary_report(self, statistics: dict, insights: list) -> str:
        """
        Create text summary report
        
        Args:
            statistics (dict): Statistical summary
            insights (list): List of insights
            
        Returns:
            str: Formatted report text
        """
        report = f"""
{'='*80}
DATA ANALYSIS REPORT
{'='*80}

Report Title: {self.title}
Generated: {self.timestamp}
Total Records: {len(self.df)}
Total Columns: {len(self.df.columns)}

{'='*80}
DATASET OVERVIEW
{'='*80}

Columns: {', '.join(self.df.columns.tolist())}

Data Types:
{self.df.dtypes.to_string()}

{'='*80}
STATISTICAL SUMMARY
{'='*80}

{str(statistics) if statistics else 'No statistics available'}

{'='*80}
KEY INSIGHTS & RECOMMENDATIONS
{'='*80}

"""
        
        for idx, insight in enumerate(insights, 1):
            report += f"{idx}. {insight}\n"
        
        report += f"\n{'='*80}\nEnd of Report\n{'='*80}"
        
        return report
