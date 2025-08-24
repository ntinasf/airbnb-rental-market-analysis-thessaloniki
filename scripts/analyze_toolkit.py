"""
Analyze Toolkit - Statistical Analysis & Pattern Recognition
Author: Data Science Toolkit
Description: Comprehensive functions for statistical analysis, hypothesis testing, and pattern detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, 
                        f_oneway, kruskal, chi2_contingency, pearsonr, 
                        spearmanr, normaltest, shapiro)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
from typing import Union, List, Dict, Tuple, Optional
from dataclasses import dataclass
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set visualization defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


@dataclass
class StatisticalTestResult:
    """Store results of statistical tests in a structured format."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None
    assumptions_met: Optional[Dict] = None


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis toolkit for data exploration.
    Handles univariate, bivariate, and multivariate analyses.
    """
    
    def __init__(self, df: pd.DataFrame, significance_level: float = 0.05):
        """
        Initialize analyzer with DataFrame.
        
        Args:
            df: Input DataFrame
            significance_level: Alpha level for hypothesis tests (default 0.05)
        """
        self.df = df.copy()
        self.alpha = significance_level
        self.results_log = []
        self._categorical_numeric_columns = None  # Cache for detected categorical numerics
        
    def detect_categorical_numerics(self, categorical_threshold: int = 10,
                                  include_binary: bool = True) -> List[str]:
        """
        Detect numeric columns that should be treated as categorical.
        
        Args:
            categorical_threshold: Max unique values to consider categorical
            include_binary: Whether to include binary (0/1) columns
            
        Returns:
            List of column names that are numeric but should be treated as categorical
        """
        if self._categorical_numeric_columns is not None:
            return self._categorical_numeric_columns
        
        categorical_numerics = []
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
                
            unique_count = data.nunique()
            unique_values = set(data.unique())
            
            # Check various conditions
            is_categorical = False
            reason = ""
            
            # Binary variables
            if include_binary and unique_count == 2:
                if unique_values.issubset({0, 1, -1, 1, True, False}):
                    is_categorical = True
                    reason = "binary"
            
            # Low cardinality
            elif unique_count <= categorical_threshold:
                is_categorical = True
                reason = f"low cardinality ({unique_count} unique values)"
            
            # Integer sequences that might be categories
            elif (data.dtype.kind == 'i' and 
                  unique_count <= 20 and 
                  data.min() >= 0 and 
                  data.max() <= 100):
                # Check if it's a sequence like 1,2,3,4,5 (ratings)
                sorted_uniques = sorted(unique_values)
                if len(sorted_uniques) > 1:
                    gaps = [sorted_uniques[i+1] - sorted_uniques[i] 
                           for i in range(len(sorted_uniques)-1)]
                    if all(gap == 1 for gap in gaps):
                        is_categorical = True
                        reason = "sequential integers (likely ordinal)"
            
            if pd.api.types.is_string_dtype(col):
            # Column name heuristics
                col_lower = col.lower()
            categorical_keywords = ['category', 'class', 'group', 'type', 'cluster',
                                  'segment', 'level', 'year', 'month', 'day', 'hour', 'quarter']
            if any(keyword in col_lower for keyword in categorical_keywords):
                is_categorical = True
                reason = f"column name suggests categorical"
            
            if is_categorical:
                categorical_numerics.append(col)
                print(f"Detected '{col}' as categorical numeric ({reason})")
        
        self._categorical_numeric_columns = categorical_numerics
        return categorical_numerics
    
    def is_numeric_categorical(self, column: str, 
                             categorical_threshold: int = 10) -> bool:
        """
        Check if a specific numeric column should be treated as categorical.
        
        Args:
            column: Column name to check
            categorical_threshold: Max unique values to consider categorical
            
        Returns:
            True if the column should be treated as categorical
        """
        if column not in self.df.columns:
            return False
            
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return False
        
        categorical_cols = self.detect_categorical_numerics(categorical_threshold)
        return column in categorical_cols
        self._categorical_numeric_columns = None  # Cache for detected categorical numerics
        
    def univariate_analysis(self, column: str, 
                          show_plots: bool = True,
                          force_categorical: bool = False,
                          categorical_threshold: int = 10) -> Dict:
        """
        Comprehensive univariate analysis for a single column.
        
        Args:
            column: Column name to analyze
            show_plots: Whether to display visualizations
            force_categorical: Force treatment as categorical regardless of type
            categorical_threshold: Max unique values to auto-treat numeric as categorical
            
        Returns:
            Dictionary with statistical summaries
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column].dropna()
        results = {'column': column, 'original_dtype': str(data.dtype)}
        
        # Determine if numeric variable should be treated as categorical
        treat_as_categorical = force_categorical
        
        if pd.api.types.is_numeric_dtype(data) and not force_categorical:
            unique_count = data.nunique()
            unique_ratio = unique_count / len(data)
            
            # Heuristics for detecting categorical numeric variables
            # 1. Low cardinality (few unique values)
            # 2. All integers (often indicates categories)
            # 3. Common patterns (binary 0/1, ratings 1-5, years)
            is_likely_categorical = (
                unique_count <= categorical_threshold or
                (data.dtype in ['int64', 'int32', 'int16', 'int8'] and unique_count <= 20) or
                (unique_count == 2 and set(data.unique()).issubset({0, 1, -1, 1})) or  # Binary
                (unique_count <= 10 and data.min() >= 0 and data.max() <= 10 and data.dtype.kind == 'i') or  # Ratings
                (column.lower() in ['year', 'month', 'day', 'hour', 'category', 'class', 'group', 'type'])  # Common categorical names
            )
            
            if is_likely_categorical:
                treat_as_categorical = True
                results['analysis_note'] = f"Numeric variable treated as categorical (unique values: {unique_count})"
        
        results['analysis_type'] = 'categorical' if treat_as_categorical else 'numeric'
        
        if not treat_as_categorical and pd.api.types.is_numeric_dtype(data):
            # Numeric analysis
            results.update({
                'count': len(data),
                'unique_values': data.nunique(),
                'mean': data.mean(),
                'median': data.median(),
                'mode': data.mode().values[0] if len(data.mode()) > 0 else None,
                'std': data.std(),
                'variance': data.var(),
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min(),
                'iqr': data.quantile(0.75) - data.quantile(0.25),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'cv': data.std() / data.mean() if data.mean() != 0 else np.nan,  # Coefficient of variation
                'percentiles': {
                    '1%': data.quantile(0.01),
                    '5%': data.quantile(0.05),
                    '25%': data.quantile(0.25),
                    '50%': data.quantile(0.50),
                    '75%': data.quantile(0.75),
                    '95%': data.quantile(0.95),
                    '99%': data.quantile(0.99)
                }
            })
            
            # Normality test
            if len(data) > 30:
                stat, p_value = normaltest(data)
                results['normality_test'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > self.alpha
                }
            
            if show_plots:
                self._plot_numeric_univariate(data, column)
                
        else:
            # Categorical analysis (including numeric treated as categorical)
            value_counts = data.value_counts()
            results.update({
                'count': len(data),
                'unique_values': data.nunique(),
                'mode': value_counts.index[0],
                'mode_frequency': value_counts.iloc[0],
                'mode_percentage': (value_counts.iloc[0] / len(data)) * 100,
                'value_counts': value_counts.to_dict(),
                'proportions': (value_counts / len(data)).to_dict(),
                'entropy': -sum((value_counts / len(data)) * np.log2(value_counts / len(data) + 1e-15)),  # Add small epsilon to avoid log(0)
                'gini_impurity': 1 - sum((value_counts / len(data)) ** 2)
            })
            
            # Additional stats for ordered categorical (if numeric)
            if pd.api.types.is_numeric_dtype(data):
                results['ordered_stats'] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'range': [data.min(), data.max()]
                }
            
            if show_plots:
                self._plot_categorical_univariate(data, column)
        
        return results
    
    def _plot_numeric_univariate(self, data: pd.Series, column: str):
        """Create comprehensive visualizations for numeric variable."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Univariate Analysis: {column}', fontsize=16)
        
        # 1. Histogram with KDE
        axes[0, 0].hist(data, bins=30, density=True, alpha=0.7, 
                       color='skyblue', edgecolor='black')
        data.plot.kde(ax=axes[0, 0], color='red', linewidth=2)
        axes[0, 0].axvline(data.mean(), color='green', linestyle='--', 
                          linewidth=2, label=f'Mean: {data.mean():.2f}')
        axes[0, 0].axvline(data.median(), color='orange', linestyle='--', 
                          linewidth=2, label=f'Median: {data.median():.2f}')
        axes[0, 0].set_title('Distribution with KDE')
        axes[0, 0].legend()
        
        # 2. Box plot
        box_data = axes[0, 1].boxplot(data, vert=True, patch_artist=True)
        box_data['boxes'][0].set_facecolor('lightblue')
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel(column)
        
        # 3. Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')
        
        # 4. Violin plot with quartiles
        parts = axes[1, 1].violinplot([data], positions=[1], showmeans=True, 
                                     showextrema=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightgreen')
            pc.set_alpha(0.7)
        axes[1, 1].set_title('Violin Plot')
        axes[1, 1].set_xticks([1])
        axes[1, 1].set_xticklabels([column])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_categorical_univariate(self, data: pd.Series, column: str):
        """Create visualizations for categorical variable."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Univariate Analysis: {column}', fontsize=16)
        
        # 1. Bar chart
        value_counts = data.value_counts()
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
        
        value_counts.plot(kind='bar', ax=axes[0], color='coral', alpha=0.8, rot=40)
        axes[0].set_title('Frequency Distribution')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Count')
        
        # 2. Pie chart (only if few categories)
        if len(value_counts) <= 6:
            value_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                            colors=sns.color_palette('pastel'))
            axes[1].set_title('Proportion Distribution')
            axes[1].set_ylabel('')
        else:
            # Treemap alternative for many categories
            axes[1].text(0.5, 0.5, f'Too many categories ({len(value_counts)}) for pie chart',
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('Proportion Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def bivariate_analysis(self, col1: str, col2: str, 
                         show_plots: bool = True,
                         categorical_threshold: int = 10) -> StatisticalTestResult:
        """
        Perform bivariate analysis based on variable types.
        
        Args:
            col1: First column name
            col2: Second column name
            show_plots: Whether to display visualizations
            categorical_threshold: Threshold for treating numeric as categorical
            
        Returns:
            StatisticalTestResult object
        """
        if col1 not in self.df.columns or col2 not in self.df.columns:
            raise ValueError("One or both columns not found")
        
        # Determine actual variable types (considering numeric categoricals)
        is_numeric1 = (pd.api.types.is_numeric_dtype(self.df[col1]) and 
                      not col1.value_counts().shape[0] < categorical_threshold)
        is_numeric2 = (pd.api.types.is_numeric_dtype(self.df[col2]) and 
                      not col1.value_counts().shape[0] < categorical_threshold))
        
        # Log the analysis type
        type1 = "numeric" if is_numeric1 else "categorical"
        type2 = "numeric" if is_numeric2 else "categorical"
        print(f"Analyzing {col1} ({type1}) vs {col2} ({type2})")
        
        if is_numeric1 and is_numeric2:
            result = self._analyze_numeric_numeric(col1, col2, show_plots)
        elif is_numeric1 != is_numeric2:
            # One numeric, one categorical
            numeric_col = col1 if is_numeric1 else col2
            cat_col = col2 if is_numeric1 else col1
            result = self._analyze_numeric_categorical(numeric_col, cat_col, show_plots)
        else:
            # Both categorical (including numeric treated as categorical)
            result = self._analyze_categorical_categorical(col1, col2, show_plots)
        
        self.results_log.append(result)
        return result
    
    def _analyze_numeric_numeric(self, col1: str, col2: str, 
                               show_plots: bool) -> StatisticalTestResult:
        """Analyze relationship between two numeric variables."""
        data1 = self.df[col1].dropna()
        data2 = self.df[col2].dropna()
        
        # Find common indices
        common_idx = data1.index.intersection(data2.index)
        data1 = data1[common_idx]
        data2 = data2[common_idx]
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(data1, data2)
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(data1, data2)
        
        # R-squared from linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)
        r_squared = r_value ** 2
        
        result = StatisticalTestResult(
            test_name="Correlation Analysis",
            statistic=pearson_r,
            p_value=pearson_p,
            effect_size=r_squared,
            interpretation=self._interpret_correlation(pearson_r, pearson_p)
        )
        
        if show_plots:
            self._plot_numeric_numeric(data1, data2, col1, col2, 
                                     slope, intercept, pearson_r, spearman_r)
        
        return result
    
    def _plot_numeric_numeric(self, data1, data2, col1, col2, 
                            slope, intercept, pearson_r, spearman_r):
        """Create scatter plot with regression line."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(data1, data2, alpha=0.6, s=50, c=data1, cmap='viridis')
        
        # Regression line
        x_line = np.array([data1.min(), data1.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
               label=f'y = {slope:.3f}x + {intercept:.3f}')
        
        # Add confidence interval
        from scipy import stats as scipy_stats
        predict_mean_se = lambda x: scipy_stats.t.ppf(1-self.alpha/2, len(data1)-2) * \
                                   np.sqrt(np.var(data2 - (slope * data1 + intercept)) * \
                                   (1/len(data1) + (x - np.mean(data1))**2 / np.sum((data1 - np.mean(data1))**2)))
        
        x_plot = np.linspace(data1.min(), data1.max(), 100)
        y_plot = slope * x_plot + intercept
        
        ax.set_xlabel(col1, fontsize=12)
        ax.set_ylabel(col2, fontsize=12)
        ax.set_title(f'{col1} vs {col2}\nPearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}', 
                    fontsize=14)
        ax.legend()
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label=col1)
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_numeric_categorical(self, numeric_col: str, cat_col: str, 
                                   show_plots: bool) -> StatisticalTestResult:
        """Analyze relationship between numeric and categorical variables."""
        groups = []
        labels = []
        
        for category in self.df[cat_col].dropna().unique():
            group_data = self.df[self.df[cat_col] == category][numeric_col].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                labels.append(str(category))
        
        n_groups = len(groups)
        
        if n_groups < 2:
            raise ValueError("Need at least 2 groups for comparison")
        
        # Choose appropriate test
        if n_groups == 2:
            # Two groups - t-test or Mann-Whitney U
            # Check normality
            normal1 = shapiro(groups[0])[1] > self.alpha if len(groups[0]) > 30 else False
            normal2 = shapiro(groups[1])[1] > self.alpha if len(groups[1]) > 30 else False
            
            if normal1 and normal2:
                # Independent t-test
                stat, p_value = ttest_ind(groups[0], groups[1])
                test_name = "Independent t-test"
                
                # Cohen's d effect size
                pooled_std = np.sqrt(((len(groups[0])-1)*groups[0].std()**2 + 
                                    (len(groups[1])-1)*groups[1].std()**2) / 
                                   (len(groups[0]) + len(groups[1]) - 2))
                effect_size = (groups[0].mean() - groups[1].mean()) / pooled_std
            else:
                # Mann-Whitney U test
                stat, p_value = mannwhitneyu(groups[0], groups[1])
                test_name = "Mann-Whitney U test"
                effect_size = stat / (len(groups[0]) * len(groups[1]))
        
        else:
            # Multiple groups - ANOVA or Kruskal-Wallis
            # Check normality for each group
            all_normal = all(shapiro(group)[1] > self.alpha 
                           for group in groups if len(group) > 30)
            
            if all_normal:
                # One-way ANOVA
                stat, p_value = f_oneway(*groups)
                test_name = "One-way ANOVA"
                
                # Eta squared effect size
                grand_mean = np.concatenate(groups).mean()
                ss_between = sum(len(group) * (group.mean() - grand_mean)**2 
                               for group in groups)
                ss_total = sum((val - grand_mean)**2 
                             for group in groups for val in group)
                effect_size = ss_between / ss_total
            else:
                # Kruskal-Wallis test
                stat, p_value = kruskal(*groups)
                test_name = "Kruskal-Wallis test"
                effect_size = (stat - n_groups + 1) / (sum(len(g) for g in groups) - n_groups)
        
        result = StatisticalTestResult(
            test_name=test_name,
            statistic=stat,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=self._interpret_group_comparison(p_value, effect_size)
        )
        
        if show_plots:
            self._plot_numeric_categorical(groups, labels, numeric_col, cat_col)
        
        return result
    
    def _plot_numeric_categorical(self, groups, labels, numeric_col, cat_col):
        """Create box plots and violin plots for group comparisons."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        bp = axes[0].boxplot(groups, labels=labels, patch_artist=True)
        colors = sns.color_palette('Set3', len(groups))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[0].set_xlabel(cat_col)
        axes[0].set_ylabel(numeric_col)
        axes[0].set_title(f'{numeric_col} by {cat_col} - Box Plot')
        
        # Violin plot with individual points
        parts = axes[1].violinplot(groups, positions=range(len(groups)), 
                                  showmeans=True, showmedians=True)
        for i, (group, color) in enumerate(zip(groups, colors)):
            # Add jittered points
            y = group
            x = np.random.normal(i, 0.04, size=len(y))
            axes[1].scatter(x, y, alpha=0.3, s=20, color=color)
        
        axes[1].set_xticks(range(len(groups)))
        axes[1].set_xticklabels(labels)
        axes[1].set_xlabel(cat_col)
        axes[1].set_ylabel(numeric_col)
        axes[1].set_title(f'{numeric_col} by {cat_col} - Violin Plot with Points')
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_categorical_categorical(self, col1: str, col2: str, 
                                       show_plots: bool) -> StatisticalTestResult:
        """Analyze relationship between two categorical variables."""
        # Create contingency table
        contingency = pd.crosstab(self.df[col1], self.df[col2])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Cramér's V effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        result = StatisticalTestResult(
            test_name="Chi-square test of independence",
            statistic=chi2,
            p_value=p_value,
            effect_size=cramers_v,
            interpretation=self._interpret_chi_square(p_value, cramers_v)
        )
        
        if show_plots:
            self._plot_categorical_categorical(contingency, col1, col2)
        
        return result
    
    def _plot_categorical_categorical(self, contingency, col1, col2):
        """Create heatmap and stacked bar chart for categorical relationships."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap
        sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title(f'Contingency Table: {col1} vs {col2}')
        axes[0].set_xlabel(col2)
        axes[0].set_ylabel(col1)
        
        # Stacked bar chart (proportions)
        contingency_pct = contingency.div(contingency.sum(axis=1), axis=0)
        contingency_pct.plot(kind='bar', stacked=True, ax=axes[1], 
                           colormap='viridis', alpha=0.8)
        axes[1].set_title(f'Proportional Distribution')
        axes[1].set_xlabel(col1)
        axes[1].set_ylabel('Proportion')
        axes[1].legend(title=col2, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def _interpret_correlation(self, r: float, p_value: float) -> str:
        """Interpret correlation coefficient."""
        if p_value > self.alpha:
            return f"No significant correlation (p={p_value:.3f})"
        
        strength = ""
        if abs(r) < 0.3:
            strength = "weak"
        elif abs(r) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if r > 0 else "negative"
        
        return f"Significant {strength} {direction} correlation (r={r:.3f}, p={p_value:.3f})"
    
    def _interpret_group_comparison(self, p_value: float, effect_size: float) -> str:
        """Interpret group comparison results."""
        if p_value > self.alpha:
            return f"No significant difference between groups (p={p_value:.3f})"
        
        effect_magnitude = ""
        if abs(effect_size) < 0.2:
            effect_magnitude = "small"
        elif abs(effect_size) < 0.5:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
        
        return f"Significant difference with {effect_magnitude} effect size (p={p_value:.3f}, effect={effect_size:.3f})"
    
    def _interpret_chi_square(self, p_value: float, cramers_v: float) -> str:
        """Interpret chi-square test results."""
        if p_value > self.alpha:
            return f"Variables are independent (p={p_value:.3f})"
        
        association = ""
        if cramers_v < 0.1:
            association = "weak"
        elif cramers_v < 0.3:
            association = "moderate"
        else:
            association = "strong"
        
        return f"Significant {association} association (Cramér's V={cramers_v:.3f}, p={p_value:.3f})"
    
    def time_series_analysis(self, value_col: str, date_col: str,
                           freq: str = 'D') -> Dict:
        """
        Perform time series analysis with decomposition.
        
        Args:
            value_col: Column with values
            date_col: Column with dates
            freq: Frequency of time series ('D', 'W', 'M', etc.)
            
        Returns:
            Dictionary with time series components
        """
        # Prepare time series data
        ts_data = self.df[[date_col, value_col]].copy()
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        ts_data = ts_data.set_index(date_col).sort_index()
        ts_data = ts_data[value_col].resample(freq).mean()
        
        # Decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if len(ts_data) > 2 * 12:  # Need enough data for decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=12)
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            ts_data.plot(ax=axes[0], color='blue')
            axes[0].set_title('Original Time Series')
            
            decomposition.trend.plot(ax=axes[1], color='green')
            axes[1].set_title('Trend Component')
            
            decomposition.seasonal.plot(ax=axes[2], color='red')
            axes[2].set_title('Seasonal Component')
            
            decomposition.resid.plot(ax=axes[3], color='purple')
            axes[3].set_title('Residual Component')
            
            plt.tight_layout()
            plt.show()
            
            return {
                'trend': decomposition.trend.dropna().tolist(),
                'seasonal': decomposition.seasonal.dropna().tolist(),
                'residual': decomposition.resid.dropna().tolist(),
                'trend_strength': decomposition.trend.std() / ts_data.std() if ts_data.std() > 0 else 0,
                'seasonal_strength': decomposition.seasonal.std() / ts_data.std() if ts_data.std() > 0 else 0
            }
        else:
            warnings.warn("Insufficient data for seasonal decomposition")
            return {'error': 'Insufficient data for decomposition'}
    
    def perform_clustering(self, columns: List[str], 
                         method: str = 'kmeans',
                         n_clusters: int = None) -> pd.Series:
        """
        Perform clustering analysis.
        
        Args:
            columns: List of columns to use for clustering
            method: 'kmeans', 'dbscan', or 'hierarchical'
            n_clusters: Number of clusters (for kmeans)
            
        Returns:
            Series with cluster labels
        """
        # Prepare data
        data = self.df[columns].dropna()
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        if method == 'kmeans':
            if n_clusters is None:
                # Find optimal k using elbow method
                n_clusters = self._find_optimal_k(data_scaled)
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(data_scaled)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(data_scaled)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate silhouette score
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(data_scaled, labels)
            print(f"Silhouette Score: {silhouette:.3f}")
        
        # Visualize if 2D
        if len(columns) == 2:
            self._plot_clusters_2d(data, labels, columns)
        elif len(columns) > 2:
            # Use PCA for visualization
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_scaled)
            self._plot_clusters_pca(data_pca, labels, pca.explained_variance_ratio_)
        
    def get_variable_types_summary(self, categorical_threshold: int = 10) -> pd.DataFrame:
        """
        Get a summary of all variable types in the dataset.
        
        Args:
            categorical_threshold: Threshold for treating numeric as categorical
            
        Returns:
            DataFrame with variable type information
        """
        # Detect categorical numerics
        categorical_numerics = self.detect_categorical_numerics(categorical_threshold)
        
        summary_data = []
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            # Determine analysis type
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if col in categorical_numerics:
                    analysis_type = "categorical (numeric)"
                    subtype = self._get_numeric_categorical_subtype(col)
                else:
                    analysis_type = "continuous"
                    subtype = self._get_numeric_subtype(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                analysis_type = "datetime"
                subtype = "temporal"
            else:
                analysis_type = "categorical"
                subtype = self._get_categorical_subtype(col)
            
            summary_data.append({
                'column': col,
                'dtype': dtype,
                'analysis_type': analysis_type,
                'subtype': subtype,
                'unique_values': unique_count,
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2),
                'sample_values': str(self.df[col].dropna().head(3).tolist())[:50] + '...'
            })
        
        return pd.DataFrame(summary_data)
    
    def _get_numeric_categorical_subtype(self, col: str) -> str:
        """Determine subtype of numeric categorical variable."""
        data = self.df[col].dropna()
        unique_values = set(data.unique())
        
        if len(unique_values) == 2 and unique_values.issubset({0, 1}):
            return "binary"
        elif all(isinstance(x, (int, np.integer)) for x in unique_values):
            if data.min() >= 0 and data.max() <= 10:
                return "ordinal (likely rating)"
            else:
                return "discrete categories"
        else:
            return "numeric categorical"
    
    def _get_numeric_subtype(self, col: str) -> str:
        """Determine subtype of continuous numeric variable."""
        data = self.df[col].dropna()
        
        if data.min() >= 0 and data.max() <= 1:
            return "proportion"
        elif all(data >= 0):
            return "positive continuous"
        else:
            return "continuous"
    
    def _get_categorical_subtype(self, col: str) -> str:
        """Determine subtype of categorical variable."""
        unique_count = self.df[col].nunique()
        
        if unique_count == 2:
            return "binary"
        elif unique_count <= 5:
            return "low cardinality"
        elif unique_count <= 20:
            return "medium cardinality"
        else:
            return "high cardinality"
    
    def _find_optimal_k(self, data: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        inertias = []
        k_range = range(2, min(max_k + 1, len(data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.show()
        
        # Simple elbow detection (customize as needed)
        optimal_k = 3  # Default
        print(f"Suggested optimal k: {optimal_k} (verify with elbow plot)")
        
        return optimal_k
    
    def _plot_clusters_2d(self, data: pd.DataFrame, labels: np.ndarray, 
                         columns: List[str]):
        """Plot clusters for 2D data."""
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6, s=50)
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.title('Clustering Results')
        plt.colorbar(scatter, label='Cluster')
        
        # Add cluster centers if kmeans
        if hasattr(labels, 'cluster_centers_'):
            centers = labels.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, 
                       alpha=0.8, marker='x', linewidths=3)
        
        plt.show()
    
    def _plot_clusters_pca(self, data_pca: np.ndarray, labels: np.ndarray,
                          explained_var: np.ndarray):
        """Plot clusters using PCA projection."""
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6, s=50)
        plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        plt.title('Clustering Results (PCA Projection)')
        plt.colorbar(scatter, label='Cluster')
        plt.show()
    
    def multivariate_regression(self, target: str, 
                              features: List[str]) -> Dict:
        """
        Perform multiple linear regression analysis.
        
        Args:
            target: Target variable name
            features: List of feature names
            
        Returns:
            Dictionary with regression results
        """
        # Prepare data
        data = self.df[[target] + features].dropna()
        X = data[features]
        y = data[target]
        
        # Add constant
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Create results dictionary
        results = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'confidence_intervals': model.conf_int().to_dict(),
            'vif': self._calculate_vif(data[features])
        }
        
        # Print summary
        print(model.summary())
        
        # Diagnostic plots
        self._regression_diagnostics(model, y)
        
        return results
    
    def _calculate_vif(self, df: pd.DataFrame) -> Dict:
        """Calculate Variance Inflation Factor for multicollinearity."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                          for i in range(len(df.columns))]
        
        return vif_data.set_index("Variable")["VIF"].to_dict()
    
    def _regression_diagnostics(self, model, y):
        """Create diagnostic plots for regression."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # 2. Q-Q plot
        stats.probplot(model.resid, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # 3. Scale-Location
        axes[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)), alpha=0.6)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Residuals|')
        axes[1, 0].set_title('Scale-Location')
        
        # 4. Residuals vs Leverage
        from statsmodels.stats.outliers_influence import OLSInfluence
        influence = OLSInfluence(model)
        leverage = influence.hat_matrix_diag
        
        axes[1, 1].scatter(leverage, model.resid, alpha=0.6)
        axes[1, 1].set_xlabel('Leverage')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Leverage')
        
        plt.tight_layout()
        plt.show()


# Utility functions for quick analyses

def quick_correlation_matrix(df: pd.DataFrame, 
                           method: str = 'pearson',
                           min_periods: int = 30) -> pd.DataFrame:
    """
    Create correlation matrix with significance testing.
    
    Args:
        df: DataFrame with numeric columns
        method: 'pearson', 'spearman', or 'kendall'
        min_periods: Minimum number of observations required
        
    Returns:
        Correlation matrix
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr(method=method, min_periods=min_periods)
    
    # Create significance matrix
    p_matrix = pd.DataFrame(np.zeros_like(corr_matrix), 
                          index=corr_matrix.index, 
                          columns=corr_matrix.columns)
    
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            data1 = df[col1].dropna()
            data2 = df[col2].dropna()
            
            # Find common indices
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) >= min_periods:
                if method == 'pearson':
                    _, p_val = pearsonr(data1[common_idx], data2[common_idx])
                else:
                    _, p_val = spearmanr(data1[common_idx], data2[common_idx])
                
                p_matrix.loc[col1, col2] = p_val
                p_matrix.loc[col2, col1] = p_val
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
               cmap='coolwarm', center=0, ax=axes[0],
               cbar_kws={'label': f'{method.capitalize()} Correlation'})
    axes[0].set_title(f'{method.capitalize()} Correlation Matrix')
    
    # Significance heatmap
    sig_matrix = p_matrix < 0.05
    sns.heatmap(sig_matrix, mask=mask, cmap='RdYlGn', center=0.5, 
               ax=axes[1], cbar_kws={'label': 'Significant (p < 0.05)'})
    axes[1].set_title('Statistical Significance')
    
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def distribution_comparison(df: pd.DataFrame, columns: List[str], 
                          group_col: str = None) -> None:
    """
    Compare distributions of multiple columns, optionally by groups.
    
    Args:
        df: Input DataFrame
        columns: List of numeric columns to compare
        group_col: Optional grouping column
    """
    n_cols = len(columns)
    
    if group_col is None:
        # Compare distributions of different columns
        fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 8))
        
        for i, col in enumerate(columns):
            data = df[col].dropna()
            
            # Histogram
            axes[0, i].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, i].set_title(f'{col} Distribution')
            axes[0, i].set_xlabel(col)
            axes[0, i].set_ylabel('Frequency')
            
            # Box plot
            axes[1, i].boxplot(data, vert=True, patch_artist=True,
                             boxprops=dict(facecolor='lightgreen'))
            axes[1, i].set_ylabel(col)
            axes[1, i].set_title(f'{col} Box Plot')
    else:
        # Compare distributions by groups
        groups = df[group_col].dropna().unique()
        n_groups = len(groups)
        
        fig, axes = plt.subplots(n_cols, 2, figsize=(12, 4*n_cols))
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(columns):
            # Density plots by group
            for group in groups:
                group_data = df[df[group_col] == group][col].dropna()
                if len(group_data) > 0:
                    group_data.plot.kde(ax=axes[i, 0], label=str(group), alpha=0.7)
            
            axes[i, 0].set_title(f'{col} Distribution by {group_col}')
            axes[i, 0].set_xlabel(col)
            axes[i, 0].legend()
            
            # Box plots by group
            group_data_list = [df[df[group_col] == g][col].dropna() for g in groups]
            bp = axes[i, 1].boxplot(group_data_list, labels=groups, patch_artist=True)
            
            colors = sns.color_palette('Set2', n_groups)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[i, 1].set_title(f'{col} by {group_col}')
            axes[i, 1].set_xlabel(group_col)
            axes[i, 1].set_ylabel(col)
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example: Create sample data with numeric categorical variables
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'continuous_var': np.random.normal(100, 15, 1000),
        'rating': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.1, 0.2, 0.3, 0.3, 0.1]),  # Ordinal
        'binary_numeric': np.random.choice([0, 1], 1000, p=[0.4, 0.6]),  # Binary as 0/1
        'category_id': np.random.choice([10, 20, 30, 40], 1000),  # Categorical IDs
        'text_category': np.random.choice(['A', 'B', 'C'], 1000),
        'year': np.random.choice([2020, 2021, 2022, 2023], 1000),  # Year as categorical
        'score': np.random.gamma(2, 2, 1000).clip(0, 10).astype(int),  # Score 0-10
        'high_card_numeric': np.random.normal(50, 10, 1000)  # True continuous
    })
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(sample_data)
    
    # Detect categorical numeric variables
    print("Detecting categorical numeric variables...")
    categorical_numerics = analyzer.detect_categorical_numerics()
    print(f"\nFound {len(categorical_numerics)} categorical numeric variables:")
    print(categorical_numerics)
    
    # Get variable types summary
    print("\nVariable Types Summary:")
    summary_df = analyzer.get_variable_types_summary()
    print(summary_df.to_string(index=False))
    
    # Univariate analysis examples
    print("\n" + "="*50)
    print("UNIVARIATE ANALYSIS EXAMPLES")
    print("="*50)
    
    # Analyze rating (numeric but categorical)
    print("\nAnalyzing 'rating' (numeric treated as categorical):")
    rating_results = analyzer.univariate_analysis('rating', show_plots=False)
    print(f"Analysis type: {rating_results['analysis_type']}")
    print(f"Unique values: {rating_results['unique_values']}")
    if 'value_counts' in rating_results:
        print(f"Value distribution: {rating_results['value_counts']}")
    
    # Analyze continuous variable
    print("\nAnalyzing 'continuous_var' (true continuous):")
    cont_results = analyzer.univariate_analysis('continuous_var', show_plots=False)
    print(f"Analysis type: {cont_results['analysis_type']}")
    print(f"Mean: {cont_results['mean']:.2f}, Std: {cont_results['std']:.2f}")
    
    # Bivariate analysis examples
    print("\n" + "="*50)
    print("BIVARIATE ANALYSIS EXAMPLES")
    print("="*50)
    
    # This should detect rating as categorical and use appropriate test
    result = analyzer.bivariate_analysis('rating', 'continuous_var', show_plots=False)
    print(f"\nTest used: {result.test_name}")
    print(f"P-value: {result.p_value:.4f}")
    
    print("\nAnalyze toolkit loaded successfully!")