"""
Share Toolkit - Data Visualization & Storytelling
Author: Data Science Toolkit
Description: Advanced visualization and presentation tools for effective data communication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from typing import Union, List, Dict, Tuple, Optional, Any
import warnings
from datetime import datetime
import textwrap

# Professional color palettes
PALETTE_CATEGORICAL = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
PALETTE_SEQUENTIAL = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
PALETTE_DIVERGING = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
PALETTE_COLORBLIND = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class DataStoryTeller:
    """
    Advanced visualization and storytelling toolkit for data presentation.
    Creates publication-ready charts with proper design principles.
    """
    
    def __init__(self, style: str = 'professional'):
        """
        Initialize with visualization style.
        
        Args:
            style: 'professional', 'minimal', 'colorful', or 'dark'
        """
        self.style = style
        self._set_style()
        self.story_elements = []
        
    def _set_style(self):
        """Apply the selected visualization style."""
        if self.style == 'professional':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = PALETTE_CATEGORICAL
        elif self.style == 'minimal':
            plt.style.use('seaborn-v0_8-white')
            self.colors = ['#333333', '#666666', '#999999', '#CCCCCC']
        elif self.style == 'colorful':
            plt.style.use('seaborn-v0_8-bright')
            self.colors = sns.color_palette('husl', 8)
        elif self.style == 'dark':
            plt.style.use('dark_background')
            self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#5D5D5D']
    
    def create_executive_dashboard(self, data: Dict[str, Any], 
                                 title: str = "Executive Dashboard") -> plt.Figure:
        """
        Create a comprehensive executive dashboard with multiple KPIs.
        
        Args:
            data: Dictionary containing:
                - 'kpis': Dict of KPI names and values
                - 'trends': DataFrame with time series data
                - 'breakdown': Dict for categorical breakdown
                - 'comparison': Dict for comparisons
            title: Dashboard title
            
        Returns:
            Figure object
        """
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
        
        # KPI Cards (top row)
        if 'kpis' in data:
            for i, (kpi_name, kpi_data) in enumerate(list(data['kpis'].items())[:4]):
                ax = fig.add_subplot(gs[0, i])
                self._create_kpi_card(ax, kpi_name, kpi_data)
        
        # Main trend chart (middle section)
        if 'trends' in data:
            ax_trend = fig.add_subplot(gs[1:3, :2])
            self._create_trend_chart(ax_trend, data['trends'])
        
        # Breakdown chart (right middle)
        if 'breakdown' in data:
            ax_breakdown = fig.add_subplot(gs[1:3, 2:])
            self._create_breakdown_chart(ax_breakdown, data['breakdown'])
        
        # Comparison chart (bottom)
        if 'comparison' in data:
            ax_comparison = fig.add_subplot(gs[3, :])
            self._create_comparison_chart(ax_comparison, data['comparison'])
        
        # Add timestamp
        fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                ha='right', va='bottom', fontsize=8, alpha=0.5)
        
        return fig
    
    def _create_kpi_card(self, ax: plt.Axes, name: str, data: Dict):
        """Create a KPI card with value and change indicator."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Background
        fancy_box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='white',
                                  edgecolor='lightgray',
                                  linewidth=2)
        ax.add_patch(fancy_box)
        
        # KPI Name
        ax.text(0.5, 0.75, name, ha='center', va='center',
               fontsize=12, color='gray')
        
        # Value
        value = data.get('value', 0)
        ax.text(0.5, 0.45, f"{value:,.0f}" if isinstance(value, (int, float)) else str(value),
               ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Change indicator
        if 'change' in data:
            change = data['change']
            color = '#2ecc71' if change >= 0 else '#e74c3c'
            arrow = '↑' if change >= 0 else '↓'
            ax.text(0.5, 0.2, f"{arrow} {abs(change):.1f}%",
                   ha='center', va='center', fontsize=10, color=color)
    
    def _create_trend_chart(self, ax: plt.Axes, df: pd.DataFrame):
        """Create a professional trend line chart."""
        # Assuming df has date index and multiple columns
        for i, col in enumerate(df.columns[:3]):  # Max 3 lines
            ax.plot(df.index, df[col], linewidth=2.5, 
                   color=self.colors[i], label=col, marker='o', 
                   markersize=4, markevery=max(len(df)//10, 1))
        
        ax.set_title('Trend Analysis', fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add subtle fill under the main line
        if len(df.columns) > 0:
            ax.fill_between(df.index, 0, df.iloc[:, 0], 
                          alpha=0.1, color=self.colors[0])
    
    def _create_breakdown_chart(self, ax: plt.Axes, data: Dict):
        """Create a donut chart for breakdown visualization."""
        values = list(data.values())
        labels = list(data.keys())
        
        # Create donut
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                         colors=self.colors[:len(values)],
                                         startangle=90, pctdistance=0.85)
        
        # Make it a donut
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        
        # Beautify
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        ax.set_title('Category Breakdown', fontsize=14, fontweight='bold', pad=20)
    
    def _create_comparison_chart(self, ax: plt.Axes, data: Dict):
        """Create a horizontal bar chart for comparisons."""
        items = list(data.keys())
        values = list(data.values())
        
        y_pos = np.arange(len(items))
        bars = ax.barh(y_pos, values, color=self.colors[0], alpha=0.8)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:,.0f}', ha='left', va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(items)
        ax.set_xlabel('Value')
        ax.set_title('Comparative Analysis', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, axis='x', alpha=0.3)
    
    def create_storytelling_slide(self, data: pd.DataFrame, 
                                story_point: str,
                                chart_type: str = 'auto',
                                highlight: Any = None) -> plt.Figure:
        """
        Create a single slide optimized for storytelling.
        
        Args:
            data: Data to visualize
            story_point: The main message/insight
            chart_type: Type of chart ('line', 'bar', 'scatter', 'auto')
            highlight: Data point or range to highlight
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Add story point as title
        wrapped_title = '\n'.join(textwrap.wrap(story_point, 60))
        fig.suptitle(wrapped_title, fontsize=16, fontweight='bold', y=0.95)
        
        # Auto-detect chart type if needed
        if chart_type == 'auto':
            if isinstance(data.index, pd.DatetimeIndex):
                chart_type = 'line'
            elif data.shape[1] == 2 and all(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns):
                chart_type = 'scatter'
            else:
                chart_type = 'bar'
        
        # Create appropriate chart
        if chart_type == 'line':
            for col in data.columns:
                ax.plot(data.index, data[col], linewidth=2.5, label=col)
            
            # Highlight specific area
            if highlight:
                if isinstance(highlight, tuple):
                    ax.axvspan(highlight[0], highlight[1], alpha=0.2, color='yellow')
                else:
                    ax.axvline(highlight, color='red', linestyle='--', linewidth=2)
        
        elif chart_type == 'bar':
            data.plot(kind='bar', ax=ax, color=self.colors[:len(data.columns)])
            
            # Highlight specific bars
            if highlight is not None:
                bars = ax.patches
                for i, bar in enumerate(bars):
                    if i in (highlight if isinstance(highlight, list) else [highlight]):
                        bar.set_color('#e74c3c')
                        bar.set_alpha(1.0)
                    else:
                        bar.set_alpha(0.6)
        
        elif chart_type == 'scatter':
            ax.scatter(data.iloc[:, 0], data.iloc[:, 1], 
                      alpha=0.6, s=100, c=self.colors[0])
            
            # Add trend line
            z = np.polyfit(data.iloc[:, 0], data.iloc[:, 1], 1)
            p = np.poly1d(z)
            ax.plot(data.iloc[:, 0], p(data.iloc[:, 0]), 
                   "r--", linewidth=2, alpha=0.8)
        
        # Clean up
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        if ax.get_legend():
            ax.legend(frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        return fig
    
    def create_comparison_visualization(self, 
                                      df: pd.DataFrame,
                                      compare_col: str,
                                      value_cols: List[str],
                                      viz_type: str = 'grouped_bar') -> plt.Figure:
        """
        Create comparison visualizations with multiple options.
        
        Args:
            df: DataFrame with data
            compare_col: Column to compare across
            value_cols: Value columns to compare
            viz_type: 'grouped_bar', 'stacked_bar', 'radar', 'parallel'
            
        Returns:
            Figure object
        """
        if viz_type == 'grouped_bar':
            return self._grouped_bar_comparison(df, compare_col, value_cols)
        elif viz_type == 'stacked_bar':
            return self._stacked_bar_comparison(df, compare_col, value_cols)
        elif viz_type == 'radar':
            return self._radar_comparison(df, compare_col, value_cols)
        elif viz_type == 'parallel':
            return self._parallel_coordinates(df, compare_col, value_cols)
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")
    
    def _grouped_bar_comparison(self, df: pd.DataFrame, 
                              compare_col: str, 
                              value_cols: List[str]) -> plt.Figure:
        """Create grouped bar chart comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        x = np.arange(len(df[compare_col].unique()))
        width = 0.8 / len(value_cols)
        
        for i, col in enumerate(value_cols):
            values = df.groupby(compare_col)[col].mean()
            offset = width * (i - len(value_cols)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, 
                         label=col, color=self.colors[i % len(self.colors)])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel(compare_col)
        ax.set_ylabel('Value')
        ax.set_title(f'Comparison of {", ".join(value_cols)} by {compare_col}')
        ax.set_xticks(x)
        ax.set_xticklabels(df[compare_col].unique())
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        return fig
    
    def _radar_comparison(self, df: pd.DataFrame, 
                        compare_col: str,
                        value_cols: List[str]) -> plt.Figure:
        """Create radar chart for multi-dimensional comparison."""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Prepare data
        categories = df[compare_col].unique()[:5]  # Limit to 5 for clarity
        angles = np.linspace(0, 2 * np.pi, len(value_cols), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, category in enumerate(categories):
            values = df[df[compare_col] == category][value_cols].mean().tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=str(category), color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(value_cols)
        ax.set_ylim(0, None)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title(f'Radar Chart Comparison by {compare_col}', size=16, y=1.08)
        
        return fig
    
    def create_annotated_visualization(self, 
                                     df: pd.DataFrame,
                                     x_col: str,
                                     y_col: str,
                                     annotations: List[Dict]) -> plt.Figure:
        """
        Create a chart with professional annotations.
        
        Args:
            df: DataFrame with data
            x_col: X-axis column
            y_col: Y-axis column
            annotations: List of dicts with 'point', 'text', 'position'
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create base plot
        ax.plot(df[x_col], df[y_col], linewidth=2.5, color=self.colors[0])
        
        # Add annotations
        for ann in annotations:
            point = ann['point']
            text = ann['text']
            position = ann.get('position', (point[0], point[1] + 0.1))
            
            # Add annotation with arrow
            ax.annotate(text, xy=point, xytext=position,
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle='arc3,rad=0.3',
                                     color='gray', lw=1.5),
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='white', 
                               edgecolor='gray',
                               alpha=0.8),
                       fontsize=10)
            
            # Mark the point
            ax.scatter(point[0], point[1], color='red', s=100, zorder=5)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{y_col} vs {x_col} with Key Insights')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_report_template(self, 
                             analyses: List[Dict],
                             title: str = "Data Analysis Report") -> List[plt.Figure]:
        """
        Create a multi-page report template with consistent styling.
        
        Args:
            analyses: List of analysis dictionaries containing:
                     - 'title': Section title
                     - 'data': Data to visualize
                     - 'viz_type': Visualization type
                     - 'insights': Key insights list
            title: Report title
            
        Returns:
            List of figure objects (pages)
        """
        pages = []
        
        # Cover page
        cover_fig = self._create_cover_page(title)
        pages.append(cover_fig)
        
        # Executive summary
        if len(analyses) > 0:
            summary_fig = self._create_executive_summary(analyses)
            pages.append(summary_fig)
        
        # Analysis pages
        for i, analysis in enumerate(analyses):
            analysis_fig = self._create_analysis_page(analysis, i+1, len(analyses))
            pages.append(analysis_fig)
        
        return pages
    
    def _create_cover_page(self, title: str) -> plt.Figure:
        """Create report cover page."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('white')
        
        # Remove axes
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.7, title, ha='center', va='center',
               fontsize=28, fontweight='bold', wrap=True)
        
        # Subtitle
        ax.text(0.5, 0.6, 'Data Analysis Report', ha='center', va='center',
               fontsize=16, color='gray')
        
        # Date
        ax.text(0.5, 0.5, datetime.now().strftime("%B %d, %Y"),
               ha='center', va='center', fontsize=14)
        
        # Add decorative line
        ax.plot([0.2, 0.8], [0.45, 0.45], 'k-', linewidth=2)
        
        # Footer
        ax.text(0.5, 0.1, 'Confidential - Internal Use Only',
               ha='center', va='center', fontsize=10, style='italic', color='gray')
        
        return fig
    
    def _create_executive_summary(self, analyses: List[Dict]) -> plt.Figure:
        """Create executive summary page."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Executive Summary', fontsize=20, fontweight='bold')
        
        # Create grid for summary boxes
        gs = GridSpec(len(analyses), 1, figure=fig, hspace=0.3)
        
        for i, analysis in enumerate(analyses):
            ax = fig.add_subplot(gs[i, 0])
            ax.axis('off')
            
            # Summary box
            box = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor='lightblue',
                               edgecolor='darkblue',
                               alpha=0.3)
            ax.add_patch(box)
            
            # Title
            ax.text(0.1, 0.65, analysis['title'], fontsize=14, 
                   fontweight='bold', va='top')
            
            # Key insight
            if 'insights' in analysis and len(analysis['insights']) > 0:
                insight = analysis['insights'][0]
                wrapped = '\n'.join(textwrap.wrap(insight, 80))
                ax.text(0.1, 0.45, wrapped, fontsize=11, va='top')
        
        return fig
    
    def create_infographic(self, 
                         data: Dict[str, Any],
                         style: str = 'modern') -> plt.Figure:
        """
        Create an infographic-style visualization.
        
        Args:
            data: Dictionary with visualization data
            style: 'modern', 'classic', or 'minimal'
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(10, 14))
        fig.patch.set_facecolor('#f8f9fa' if style != 'modern' else '#2c3e50')
        
        # Create main grid
        gs = GridSpec(6, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title section
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        title_color = '#2c3e50' if style != 'modern' else 'white'
        ax_title.text(0.5, 0.5, data.get('title', 'Data Insights'),
                     ha='center', va='center', fontsize=24, 
                     fontweight='bold', color=title_color)
        
        # Create various infographic elements
        # This is customizable based on data structure
        element_positions = [
            (1, 0), (1, 1),  # Row 2
            (2, 0), (2, 1),  # Row 3
            (3, :),          # Row 4 (full width)
            (4, 0), (4, 1),  # Row 5
            (5, :)           # Row 6 (full width)
        ]
        
        # Add various visualization elements based on data
        if 'stats' in data:
            ax = fig.add_subplot(gs[1, 0])
            self._create_stat_box(ax, data['stats'], style)
        
        if 'trend' in data:
            ax = fig.add_subplot(gs[3, :])
            self._create_mini_trend(ax, data['trend'], style)
        
        if 'comparison' in data:
            ax = fig.add_subplot(gs[2, :])
            self._create_icon_comparison(ax, data['comparison'], style)
        
        return fig
    
    def _create_stat_box(self, ax: plt.Axes, stats: Dict, style: str):
        """Create a statistics box for infographic."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Background
        bg_color = 'white' if style != 'modern' else '#34495e'
        text_color = '#2c3e50' if style != 'modern' else 'white'
        
        box = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                           boxstyle="round,pad=0.05",
                           facecolor=bg_color,
                           edgecolor='none' if style == 'modern' else 'lightgray',
                           linewidth=2)
        ax.add_patch(box)
        
        # Add statistics
        y_pos = 0.7
        for key, value in list(stats.items())[:3]:
            ax.text(0.5, y_pos, key, ha='center', fontsize=10, 
                   color=text_color, alpha=0.7)
            ax.text(0.5, y_pos - 0.15, str(value), ha='center', 
                   fontsize=16, fontweight='bold', color=text_color)
            y_pos -= 0.3


# Specialized visualization functions

def create_waterfall_chart(df: pd.DataFrame, 
                         values_col: str,
                         categories_col: str,
                         title: str = "Waterfall Chart") -> plt.Figure:
    """
    Create a waterfall chart showing cumulative effect of sequential values.
    
    Args:
        df: DataFrame with data
        values_col: Column with values
        categories_col: Column with category labels
        title: Chart title
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate cumulative values
    cumulative = df[values_col].cumsum()
    
    # Create the waterfall
    for i, (idx, row) in enumerate(df.iterrows()):
        value = row[values_col]
        category = row[categories_col]
        
        # Determine color
        if i == 0:
            color = PALETTE_CATEGORICAL[0]
            bottom = 0
        elif i == len(df) - 1:
            color = PALETTE_CATEGORICAL[2]
            bottom = 0
            value = cumulative.iloc[-1]
        else:
            color = PALETTE_CATEGORICAL[1] if value >= 0 else PALETTE_CATEGORICAL[3]
            bottom = cumulative.iloc[i-1]
        
        # Draw bar
        bar = ax.bar(i, abs(value), bottom=bottom, color=color, alpha=0.8)
        
        # Add connector line
        if i > 0 and i < len(df) - 1:
            ax.plot([i-1+0.4, i-0.4], [cumulative.iloc[i-1], cumulative.iloc[i-1]], 
                   'k--', alpha=0.5)
        
        # Add value label
        label_y = bottom + value/2 if value >= 0 else bottom + value/2
        ax.text(i, label_y, f'{value:+.0f}', ha='center', va='center', 
               fontweight='bold', fontsize=10)
    
    # Customize
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df[categories_col], rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add baseline
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    return fig


def create_funnel_chart(stages: List[str], 
                       values: List[float],
                       title: str = "Conversion Funnel") -> plt.Figure:
    """
    Create a funnel chart for conversion analysis.
    
    Args:
        stages: List of stage names
        values: List of values for each stage
        title: Chart title
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize values
    max_value = max(values)
    normalized = [v / max_value for v in values]
    
    # Create funnel
    for i, (stage, value, norm) in enumerate(zip(stages, values, normalized)):
        # Calculate trapezoid coordinates
        y_bottom = i
        y_top = i + 0.8
        x_bottom_left = 0.5 - norm/2
        x_bottom_right = 0.5 + norm/2
        
        if i < len(stages) - 1:
            next_norm = normalized[i + 1]
            x_top_left = 0.5 - next_norm/2
            x_top_right = 0.5 + next_norm/2
        else:
            x_top_left = x_bottom_left
            x_top_right = x_bottom_right
        
        # Draw trapezoid
        vertices = [(x_bottom_left, y_bottom), (x_bottom_right, y_bottom),
                   (x_top_right, y_top), (x_top_left, y_top)]
        
        from matplotlib.patches import Polygon
        trapezoid = Polygon(vertices, facecolor=PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)], 
                          alpha=0.8, edgecolor='white', linewidth=2)
        ax.add_patch(trapezoid)
        
        # Add text
        ax.text(0.5, y_bottom + 0.4, f'{stage}\n{value:,.0f}', 
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add conversion rate
        if i > 0:
            conversion_rate = (value / values[i-1]) * 100
            ax.text(0.9, y_bottom + 0.4, f'{conversion_rate:.1f}%', 
                   ha='center', va='center', fontsize=10, style='italic')
    
    # Customize
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(stages) + 0.5)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig


def create_bullet_chart(df: pd.DataFrame, 
                       metric_col: str,
                       target_col: str,
                       ranges: List[Tuple[float, float, str]],
                       title: str = "Performance Metrics") -> plt.Figure:
    """
    Create bullet charts for KPI visualization.
    
    Args:
        df: DataFrame with metrics
        metric_col: Column with actual values
        target_col: Column with target values
        ranges: List of (min, max, label) tuples for background ranges
        title: Chart title
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(len(df), 1, figsize=(10, 2*len(df)), 
                           sharex=True)
    if len(df) == 1:
        axes = [axes]
    
    for idx, (i, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        
        # Draw background ranges
        for range_min, range_max, label in ranges:
            width = range_max - range_min
            color = {'Poor': '#ffcccc', 'Fair': '#ffffcc', 'Good': '#ccffcc', 
                    'Excellent': '#ccffff'}.get(label, '#f0f0f0')
            ax.barh(0, width, left=range_min, height=0.5, 
                   color=color, alpha=0.5)
        
        # Draw actual value
        ax.barh(0, row[metric_col], height=0.3, color='black')
        
        # Draw target marker
        ax.plot([row[target_col], row[target_col]], [-0.4, 0.4], 
               'r-', linewidth=3)
        
        # Labels
        ax.text(-0.1, 0, str(i), ha='right', va='center', fontweight='bold')
        ax.text(row[metric_col] + 1, 0, f'{row[metric_col]:.0f}', 
               ha='left', va='center')
        
        # Customize
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, max(range_max for _, range_max, _ in ranges))
        ax.axis('off')
    
    # Title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Legend
    legend_elements = [mpatches.Patch(color=color, label=label, alpha=0.5)
                      for _, _, label in ranges
                      for color in [{'Poor': '#ffcccc', 'Fair': '#ffffcc', 
                                   'Good': '#ccffcc', 'Excellent': '#ccffff'}.get(label, '#f0f0f0')]]
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=len(ranges))
    
    plt.tight_layout()
    return fig


def create_heatmap_calendar(df: pd.DataFrame, 
                          date_col: str,
                          value_col: str,
                          year: int = None) -> plt.Figure:
    """
    Create a calendar heatmap visualization.
    
    Args:
        df: DataFrame with daily data
        date_col: Column with dates
        value_col: Column with values to visualize
        year: Year to visualize (default: most recent)
        
    Returns:
        Figure object
    """
    # Prepare data
    df_cal = df.copy()
    df_cal[date_col] = pd.to_datetime(df_cal[date_col])
    
    if year is None:
        year = df_cal[date_col].max().year
    
    # Filter for specified year
    df_year = df_cal[df_cal[date_col].dt.year == year].copy()
    df_year['day'] = df_year[date_col].dt.dayofyear
    df_year['week'] = df_year[date_col].dt.isocalendar().week
    df_year['weekday'] = df_year[date_col].dt.weekday
    
    # Create pivot table
    pivot = df_year.pivot_table(values=value_col, index='weekday', 
                               columns='week', aggfunc='mean')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 4))
    
    # Create heatmap
    sns.heatmap(pivot, cmap='YlOrRd', linewidths=1, linecolor='white',
               square=True, cbar_kws={'label': value_col}, ax=ax)
    
    # Customize
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_xlabel('Week of Year')
    ax.set_ylabel('Day of Week')
    ax.set_title(f'{value_col} Calendar Heatmap - {year}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add month labels
    month_starts = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='MS')
    for month_start in month_starts:
        week = month_start.isocalendar()[1]
        ax.text(week - 0.5, -0.5, month_start.strftime('%b'), 
               ha='center', va='top')
    
    plt.tight_layout()
    return fig


# Report generation utilities

class ReportGenerator:
    """Generate professional PDF-ready reports with multiple visualizations."""
    
    def __init__(self, style: str = 'professional'):
        self.storyteller = DataStoryTeller(style)
        self.figures = []
        
    def add_section(self, title: str, content: Union[plt.Figure, str, pd.DataFrame]):
        """Add a section to the report."""
        section = {
            'title': title,
            'content': content,
            'type': type(content).__name__
        }
        self.figures.append(section)
    
    def generate_report(self, filename: str = None) -> List[plt.Figure]:
        """Generate the complete report."""
        report_figures = []
        
        # Create title page
        title_fig = self._create_title_page()
        report_figures.append(title_fig)
        
        # Create table of contents
        toc_fig = self._create_table_of_contents()
        report_figures.append(toc_fig)
        
        # Add all sections
        for i, section in enumerate(self.figures):
            if isinstance(section['content'], plt.Figure):
                report_figures.append(section['content'])
            elif isinstance(section['content'], pd.DataFrame):
                table_fig = self._create_table_page(section['title'], 
                                                   section['content'])
                report_figures.append(table_fig)
            elif isinstance(section['content'], str):
                text_fig = self._create_text_page(section['title'], 
                                                section['content'])
                report_figures.append(text_fig)
        
        if filename:
            # Save as multi-page PDF
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(filename) as pdf:
                for fig in report_figures:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        
        return report_figures
    
    def _create_title_page(self) -> plt.Figure:
        """Create report title page."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Company logo placeholder
        ax.text(0.5, 0.8, '[ LOGO ]', ha='center', va='center',
               fontsize=24, color='gray', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        
        # Report title
        ax.text(0.5, 0.5, 'Data Analysis Report', ha='center', va='center',
               fontsize=28, fontweight='bold')
        
        # Date
        ax.text(0.5, 0.3, datetime.now().strftime("%B %Y"), 
               ha='center', va='center', fontsize=16)
        
        # Decorative line
        ax.plot([0.2, 0.8], [0.25, 0.25], 'k-', linewidth=2)
        
        return fig
    
    def _create_table_of_contents(self) -> plt.Figure:
        """Create table of contents page."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, 'Table of Contents', ha='center', va='center',
               fontsize=20, fontweight='bold')
        
        # Contents
        y_pos = 0.8
        for i, section in enumerate(self.figures):
            ax.text(0.2, y_pos, f"{i+1}. {section['title']}", 
                   ha='left', va='center', fontsize=12)
            ax.text(0.8, y_pos, f"Page {i+3}", 
                   ha='right', va='center', fontsize=12)
            
            # Dotted line
            ax.plot([0.35, 0.75], [y_pos, y_pos], 'k:', alpha=0.5)
            
            y_pos -= 0.05
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'revenue': np.cumsum(np.random.randn(100)) + 100,
        'users': np.cumsum(np.random.randn(100)) + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Initialize storyteller
    storyteller = DataStoryTeller(style='professional')
    
    # Create executive dashboard
    dashboard_data = {
        'kpis': {
            'Total Revenue': {'value': 15420, 'change': 12.5},
            'Active Users': {'value': 3241, 'change': -2.3},
            'Conversion Rate': {'value': '23.4%', 'change': 5.2},
            'Avg Order Value': {'value': 85.20, 'change': 8.1}
        },
        'trends': sample_data.set_index('date')[['revenue', 'users']],
        'breakdown': {'Category A': 45, 'Category B': 30, 'Category C': 25},
        'comparison': {'Q1': 12000, 'Q2': 15000, 'Q3': 18000, 'Q4': 21000}
    }
    
    # dashboard = storyteller.create_executive_dashboard(dashboard_data)
    # plt.show()
    
    print("Share toolkit loaded successfully!")
