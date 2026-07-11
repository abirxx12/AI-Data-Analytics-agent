"""
Visualization Module
Creates interactive charts and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple


class Visualizer:
    """
    Class to create various types of visualizations
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize Visualizer with dataframe
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
        """
        self.df = dataframe
        # Set dark style for professional appearance
        plt.style.use('dark_background')
        sns.set_palette("husl")
    
    def create_bar_chart(self, x_col: str, y_col: str = None, 
                        title: str = "Bar Chart", figsize: Tuple = (12, 6)):
        """
        Create bar chart
        
        Args:
            x_col (str): Column for X-axis
            y_col (str): Column for Y-axis (optional)
            title (str): Chart title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if y_col is None:
            self.df[x_col].value_counts().plot(kind='bar', ax=ax, color='steelblue')
        else:
            self.df.groupby(x_col)[y_col].sum().plot(kind='bar', ax=ax, color='steelblue')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col or 'Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def create_line_chart(self, x_col: str, y_col: str, 
                         title: str = "Line Chart", figsize: Tuple = (12, 6)):
        """
        Create line chart
        
        Args:
            x_col (str): Column for X-axis
            y_col (str): Column for Y-axis
            title (str): Chart title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.df[x_col], self.df[y_col], marker='o', linewidth=2, 
               markersize=6, color='cyan', label=y_col)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    def create_pie_chart(self, column: str, title: str = "Pie Chart", figsize: Tuple = (10, 8)):
        """
        Create pie chart
        
        Args:
            column (str): Column to visualize
            title (str): Chart title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        value_counts = self.df[column].value_counts().head(10)
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
              startangle=90, colors=colors)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        return fig
    
    def create_histogram(self, column: str, bins: int = 30, 
                        title: str = "Histogram", figsize: Tuple = (12, 6)):
        """
        Create histogram for numeric column
        
        Args:
            column (str): Column to visualize
            bins (int): Number of bins
            title (str): Chart title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(self.df[column], bins=bins, color='coral', edgecolor='black', alpha=0.7)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def create_heatmap(self, columns: List[str] = None, 
                      title: str = "Correlation Heatmap", figsize: Tuple = (10, 8)):
        """
        Create correlation heatmap
        
        Args:
            columns (List[str]): Numeric columns to include (None = all numeric)
            title (str): Chart title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if columns is None:
            numeric_df = self.df.select_dtypes(include=[np.number])
        else:
            numeric_df = self.df[columns].select_dtypes(include=[np.number])
        
        correlation = numeric_df.corr()
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   cbar=True, ax=ax, square=True, linewidths=0.5)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        return fig
    
    def create_scatter_plot(self, x_col: str, y_col: str, 
                           title: str = "Scatter Plot", color_col: str = None,
                           figsize: Tuple = (12, 6)):
        """
        Create scatter plot
        
        Args:
            x_col (str): Column for X-axis
            y_col (str): Column for Y-axis
            title (str): Chart title
            color_col (str): Column for color coding (optional)
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_col is None:
            ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6, s=50, color='cyan')
        else:
            scatter = ax.scatter(self.df[x_col], self.df[y_col], 
                               c=self.df[color_col], alpha=0.6, s=50, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label=color_col)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def create_box_plot(self, columns: List[str] = None, 
                       title: str = "Box Plot", figsize: Tuple = (12, 6)):
        """
        Create box plot for numeric columns
        
        Args:
            columns (List[str]): Columns to plot (None = all numeric)
            title (str): Chart title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.df[columns].plot(kind='box', ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def create_distribution_plot(self, column: str, 
                                title: str = "Distribution Plot", figsize: Tuple = (12, 6)):
        """
        Create distribution plot with KDE
        
        Args:
            column (str): Column to visualize
            title (str): Chart title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        self.df[column].plot(kind='hist', bins=30, alpha=0.6, ax=ax, color='skyblue')
        self.df[column].plot(kind='kde', ax=ax, secondary_y=False, color='red', linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(column, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def create_multiple_histograms(self, columns: List[str], 
                                  title: str = "Multiple Distributions", figsize: Tuple = (15, 10)):
        """
        Create multiple histograms in subplots
        
        Args:
            columns (List[str]): Columns to visualize
            title (str): Chart title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        num_cols = len(columns)
        num_rows = (num_cols + 1) // 2
        
        fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        
        for idx, col in enumerate(columns):
            if col in self.df.columns:
                axes[idx].hist(self.df[col], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
                axes[idx].set_title(col, fontsize=12)
                axes[idx].grid(axis='y', alpha=0.3)
        
        # Hide empty subplots
        for idx in range(num_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        return fig
