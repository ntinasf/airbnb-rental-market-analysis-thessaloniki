"""
Scripts package for Airbnb Thessaloniki data processing and analysis.

This package contains modules for:
- data_preprocessing: Anonymization and cleaning of listings data
- calendar_preprocessing: Processing and optimization of calendar data
- eda_functions: Reusable exploratory data analysis functions
- process_toolkit: General processing utilities
- share_toolkit: Sharing and export utilities
"""

__version__ = "0.1.0"

# Make key functions easily importable
from scripts.eda_functions import (
    analyze_numeric_variable,
    analyze_categorical_variable,
    plot_scatter,
    analyze_categorical_categorical,
    analyze_categorical_numerical,
    analyze_numerical_numerical,
)

__all__ = [
    "analyze_numeric_variable",
    "analyze_categorical_variable",
    "plot_scatter",
    "analyze_categorical_categorical",
    "analyze_categorical_numerical",
    "analyze_numerical_numerical",
]
