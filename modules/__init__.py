"""
AI Analytics Agent Modules
Collection of modules for data analysis, ML, and visualization
"""

from .auth import login_user, logout_user, check_login_status
from .data_processor import DataProcessor
from .statistical_analysis import StatisticalAnalysis
from .visualization import Visualizer
from .ml_analysis import MLAnalyzer
from .insight_generator import InsightGenerator
from .report_generator import ReportGenerator

__all__ = [
    'login_user',
    'logout_user',
    'check_login_status',
    'DataProcessor',
    'StatisticalAnalysis',
    'Visualizer',
    'MLAnalyzer',
    'InsightGenerator',
    'ReportGenerator'
]
