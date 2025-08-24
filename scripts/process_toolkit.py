"""
Process Toolkit - Data Cleaning & Validation Functions
Author: Data Science Toolkit
Description: Reusable functions for data processing, cleaning, and validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Tuple, Optional
import warnings
from datetime import datetime

# Set default style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DataProcessor:
    """
    A comprehensive class for data processing and validation tasks.
    Handles cleaning, transformation, and integrity checks.
    """
    
    def __init__(self, df: pd.DataFrame, name: str = "Dataset"):
        """
        Initialize with a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            name: Name of the dataset for reporting
        """
        self.df = df.copy()  # Always work on a copy
        self.name = name
        self.original_shape = df.shape
        self.cleaning_log = []
        
    def add_to_log(self, action: str, details: str):
        """Add action to cleaning log for documentation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cleaning_log.append({
            'timestamp': timestamp,
            'action': action,
            'details': details
        })
    
    def get_initial_summary(self) -> Dict:
        """
        Get comprehensive initial data summary.
        
        Returns:
            Dictionary with data overview statistics
        """
        summary = {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'column_types': self.df.dtypes.value_counts().to_dict(),
            'missing_summary': self.df.isnull().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        # Add numeric column statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = self.df[numeric_cols].describe().to_dict()
        
        return summary
    
    def visualize_missing_data(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create comprehensive missing data visualization.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Missing Data Analysis - {self.name}', fontsize=16)
        
        # 1. Missing data heatmap
        sns.heatmap(self.df.isnull(), 
                   cbar=True, 
                   yticklabels=False,
                   cmap='RdYlBu_r',
                   ax=axes[0, 0])
        axes[0, 0].set_title('Missing Data Patterns')
        
        # 2. Missing data bar chart
        missing_counts = self.df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
        if len(missing_counts) > 0:
            missing_counts.plot(kind='barh', ax=axes[0, 1], color='coral')
            axes[0, 1].set_title('Missing Values by Column')
            axes[0, 1].set_xlabel('Count')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Data', 
                           ha='center', va='center', fontsize=14)
            axes[0, 1].set_title('Missing Values by Column')
        
        # 3. Missing data percentage
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if len(missing_pct) > 0:
            missing_pct.plot(kind='bar', ax=axes[1, 0], color='skyblue')
            axes[1, 0].set_title('Missing Data Percentage')
            axes[1, 0].set_ylabel('Percentage (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Data', 
                           ha='center', va='center', fontsize=14)
            axes[1, 0].set_title('Missing Data Percentage')
        
        # 4. Correlation of missingness
        missing_df = self.df.isnull().astype(int)
        corr_matrix = missing_df.corr()
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   cmap='coolwarm',
                   center=0,
                   annot=True,
                   fmt='.2f',
                   ax=axes[1, 1])
        axes[1, 1].set_title('Correlation of Missing Data')
        
        plt.tight_layout()
        return fig
    
    def handle_missing_data(self, 
                           strategy: Dict[str, str] = None,
                           threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing data with customizable strategies per column type.
        
        Args:
            strategy: Dictionary mapping column names to strategies
                     Options: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 
                             'interpolate', 'drop', 'constant'
            threshold: Drop columns with more than this fraction of missing values
            
        Returns:
            Cleaned DataFrame
        """
        if strategy is None:
            # Default strategies by data type
            strategy = {}
            for col in self.df.columns:
                if self.df[col].isnull().sum() == 0:
                    continue
                    
                # Customize these defaults based on your domain
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    strategy[col] = 'median'  # Robust to outliers
                elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    strategy[col] = 'forward_fill'
                else:
                    strategy[col] = 'mode'
        
        # Drop columns exceeding threshold
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.add_to_log('Dropped columns', f"Removed {cols_to_drop} (>{threshold*100}% missing)")
        
        # Apply strategies
        for col, method in strategy.items():
            if col not in self.df.columns:
                continue
                
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if method == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif method == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif method == 'mode':
                mode_value = self.df[col].mode()
                if len(mode_value) > 0:
                    self.df[col].fillna(mode_value[0], inplace=True)
            elif method == 'forward_fill':
                self.df[col].fillna(method='ffill', inplace=True)
            elif method == 'backward_fill':
                self.df[col].fillna(method='bfill', inplace=True)
            elif method == 'interpolate':
                self.df[col].interpolate(method='linear', inplace=True)
            elif method == 'drop':
                self.df = self.df.dropna(subset=[col])
            elif isinstance(method, (int, float, str)):
                self.df[col].fillna(method, inplace=True)
            
            self.add_to_log('Missing data handled', 
                           f"{col}: {method} ({missing_count} values)")
        
        return self.df
    
    def detect_outliers(self, 
                       columns: List[str] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, pd.Series]:
        """
        Detect outliers using various methods.
        
        Args:
            columns: List of columns to check (None for all numeric)
            method: 'iqr', 'zscore', or 'isolation'
            threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)
            
        Returns:
            Dictionary mapping column names to boolean Series of outliers
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = (self.df[col] < (Q1 - threshold * IQR)) | \
                               (self.df[col] > (Q3 + threshold * IQR))
                               
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers[col] = z_scores > threshold
                
            outlier_count = outliers[col].sum()
            if outlier_count > 0:
                self.add_to_log('Outliers detected', 
                               f"{col}: {outlier_count} outliers ({method} method)")
        
        return outliers
    
    def transform_features(self, 
                          transformations: Dict[str, str]) -> pd.DataFrame:
        """
        Apply various transformations to features.
        
        Args:
            transformations: Dict mapping column names to transformation types
                           Options: 'log', 'sqrt', 'square', 'reciprocal', 'standardize', 
                                   'normalize', 'robust_scale'
        
        Returns:
            Transformed DataFrame
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        for col, transform in transformations.items():
            if col not in self.df.columns:
                warnings.warn(f"Column {col} not found in DataFrame")
                continue
            
            # Store original for comparison
            original_values = self.df[col].copy()
            
            try:
                if transform == 'log':
                    # Add small constant to avoid log(0)
                    self.df[col] = np.log1p(self.df[col])
                    
                elif transform == 'sqrt':
                    self.df[col] = np.sqrt(np.abs(self.df[col]))
                    
                elif transform == 'square':
                    self.df[col] = self.df[col] ** 2
                    
                elif transform == 'reciprocal':
                    # Avoid division by zero
                    self.df[col] = 1 / (self.df[col] + 1e-8)
                    
                elif transform == 'standardize':
                    scaler = StandardScaler()
                    self.df[col] = scaler.fit_transform(self.df[[col]])
                    
                elif transform == 'normalize':
                    scaler = MinMaxScaler()
                    self.df[col] = scaler.fit_transform(self.df[[col]])
                    
                elif transform == 'robust_scale':
                    scaler = RobustScaler()
                    self.df[col] = scaler.fit_transform(self.df[[col]])
                
                self.add_to_log('Feature transformed', f"{col}: {transform}")
                
            except Exception as e:
                warnings.warn(f"Failed to transform {col}: {str(e)}")
                self.df[col] = original_values
        
        return self.df
    
    def encode_categorical(self, 
                          columns: List[str] = None,
                          encoding_type: str = 'onehot',
                          max_categories: int = 10) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            columns: List of columns to encode (None for all categorical)
            encoding_type: 'onehot', 'label', or 'target'
            max_categories: Maximum categories for one-hot encoding
            
        Returns:
            DataFrame with encoded features
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            n_unique = self.df[col].nunique()
            
            if encoding_type == 'onehot' and n_unique <= max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
                self.add_to_log('One-hot encoded', f"{col}: {n_unique} categories")
                
            elif encoding_type == 'label' or n_unique > max_categories:
                # Label encoding
                self.df[col] = pd.Categorical(self.df[col]).codes
                self.add_to_log('Label encoded', f"{col}: {n_unique} categories")
        
        return self.df
    
    def create_time_features(self, 
                           datetime_col: str,
                           features: List[str] = None) -> pd.DataFrame:
        """
        Extract time-based features from datetime column.
        
        Args:
            datetime_col: Name of datetime column
            features: List of features to extract (default: common features)
            
        Returns:
            DataFrame with new time features
        """
        if datetime_col not in self.df.columns:
            raise ValueError(f"Column {datetime_col} not found")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.df[datetime_col]):
            self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])
        
        if features is None:
            features = ['year', 'month', 'day', 'dayofweek', 'hour', 
                       'is_weekend', 'quarter', 'is_month_start', 'is_month_end']
        
        dt_series = self.df[datetime_col]
        
        # Extract features - customize based on your needs
        feature_mapping = {
            'year': dt_series.dt.year,
            'month': dt_series.dt.month,
            'day': dt_series.dt.day,
            'dayofweek': dt_series.dt.dayofweek,
            'hour': dt_series.dt.hour,
            'minute': dt_series.dt.minute,
            'is_weekend': dt_series.dt.dayofweek.isin([5, 6]).astype(int),
            'quarter': dt_series.dt.quarter,
            'is_month_start': dt_series.dt.is_month_start.astype(int),
            'is_month_end': dt_series.dt.is_month_end.astype(int),
            'week': dt_series.dt.isocalendar().week,
            'dayofyear': dt_series.dt.dayofyear
        }
        
        for feature in features:
            if feature in feature_mapping:
                self.df[f"{datetime_col}_{feature}"] = feature_mapping[feature]
        
        self.add_to_log('Time features created', f"From {datetime_col}: {features}")
        
        return self.df
    
    def validate_data_integrity(self) -> Dict[str, List]:
        """
        Perform comprehensive data integrity checks.
        
        Returns:
            Dictionary of validation results
        """
        issues = {
            'duplicates': [],
            'type_mismatches': [],
            'range_violations': [],
            'consistency_issues': []
        }
        
        # Check for duplicate rows
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            issues['duplicates'].append(f"{dup_count} duplicate rows found")
        
        # Check for type consistency
        for col in self.df.columns:
            # Check if numeric column has non-numeric values
            if pd.api.types.is_numeric_dtype(self.df[col]):
                non_numeric = pd.to_numeric(self.df[col], errors='coerce').isnull().sum()
                if non_numeric > 0:
                    issues['type_mismatches'].append(
                        f"{col}: {non_numeric} non-numeric values in numeric column"
                    )
        
        # Check for negative values where they shouldn't be
        # Customize these based on your domain
        non_negative_cols = [col for col in self.df.columns 
                           if any(keyword in col.lower() 
                                 for keyword in ['age', 'count', 'quantity', 'price'])]
        
        for col in non_negative_cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    issues['range_violations'].append(
                        f"{col}: {negative_count} negative values"
                    )
        
        return issues
    
    def generate_cleaning_report(self, save_path: str = None) -> str:
        """
        Generate comprehensive cleaning report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        report = f"""
Data Cleaning Report - {self.name}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*50}

Original Shape: {self.original_shape}
Final Shape: {self.df.shape}
Rows Removed: {self.original_shape[0] - self.df.shape[0]}
Columns Removed: {self.original_shape[1] - self.df.shape[1]}

Cleaning Log:
{'='*50}
"""
        for entry in self.cleaning_log:
            report += f"\n[{entry['timestamp']}] {entry['action']}: {entry['details']}"
        
        report += f"\n\nFinal Data Summary:\n{'='*50}\n"
        report += f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
        report += f"Remaining Missing Values: {self.df.isnull().sum().sum()}\n"
        report += f"\nColumn Types:\n{self.df.dtypes.value_counts()}"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report


# Standalone utility functions

def quick_eda(df: pd.DataFrame, target_col: str = None) -> None:
    """
    Perform quick exploratory data analysis with visualizations.
    
    Args:
        df: Input DataFrame
        target_col: Optional target column for additional analysis
    """
    print("=== Quick EDA Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn Types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Missing data
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)[:10]
    missing_pct.plot(kind='barh', ax=axes[0, 0], color='coral')
    axes[0, 0].set_title('Top 10 Columns with Missing Data')
    axes[0, 0].set_xlabel('Percentage (%)')
    
    # 2. Numeric distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(ax=axes[0, 1], bins=20, color='skyblue', alpha=0.7)
        axes[0, 1].set_title('Numeric Column Distributions')
    
    # 3. Correlation heatmap
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Correlation Matrix')
    
    # 4. Target distribution (if provided)
    if target_col and target_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            df[target_col].hist(ax=axes[1, 1], bins=30, color='green', alpha=0.7)
        else:
            df[target_col].value_counts()[:10].plot(kind='bar', ax=axes[1, 1], color='green')
        axes[1, 1].set_title(f'Target Distribution: {target_col}')
    
    plt.tight_layout()
    plt.show()


def create_sample_splits(df: pd.DataFrame, 
                        test_size: float = 0.2,
                        stratify_col: str = None,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test splits with optional stratification.
    
    Args:
        df: Input DataFrame
        test_size: Fraction for test set
        stratify_col: Column name for stratified splitting
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    if stratify_col and stratify_col in df.columns:
        stratify = df[stratify_col]
    else:
        stratify = None
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=stratify,
        random_state=random_state
    )
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    if stratify_col:
        print(f"\nTarget distribution in train:\n{train_df[stratify_col].value_counts(normalize=True)}")
        print(f"\nTarget distribution in test:\n{test_df[stratify_col].value_counts(normalize=True)}")
    
    return train_df, test_df


# Example usage
if __name__ == "__main__":
    # Example: Load and process data
    # df = pd.read_csv('your_data.csv')
    
    # Initialize processor
    # processor = DataProcessor(df, name="Customer Data")
    
    # Get initial summary
    # summary = processor.get_initial_summary()
    # print(summary)
    
    # Visualize missing data
    # fig = processor.visualize_missing_data()
    # plt.show()
    
    # Handle missing data with custom strategy
    # strategy = {
    #     'age': 'median',
    #     'income': 'mean',
    #     'category': 'mode',
    #     'date': 'forward_fill'
    # }
    # df_cleaned = processor.handle_missing_data(strategy=strategy)
    
    # Detect outliers
    # outliers = processor.detect_outliers(method='iqr')
    
    # Transform features
    # transformations = {
    #     'income': 'log',
    #     'age': 'standardize'
    # }
    # df_transformed = processor.transform_features(transformations)
    
    # Generate report
    # report = processor.generate_cleaning_report('cleaning_report.txt')
    
    print("Process toolkit loaded successfully!")
