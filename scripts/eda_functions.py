"""
EDA Functions for Airbnb Data Analysis.

This module provides comprehensive functions for exploratory data analysis including:
- Univariate analysis for numeric and categorical variables
- Bivariate analysis for all variable type combinations
- Statistical tests and visualizations

Functions can be imported into notebooks and scripts for analysis.

Example:
    from scripts.eda_functions import analyze_numeric_variable
    analyze_numeric_variable(df['price'])
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from scipy import stats
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    ttest_ind,
    pearsonr,
    spearmanr,
    gaussian_kde,
)
from sklearn.metrics import r2_score

__all__ = [
    "analyze_numeric_variable",
    "analyze_categorical_variable",
    "plot_scatter",
    "analyze_categorical_categorical",
    "analyze_categorical_numerical",
    "analyze_numerical_numerical",
]


def analyze_numeric_variable(data: pd.Series, include_outliers: bool = True) -> None:
    """
    Analyze a numeric variable with descriptive statistics and visualizations.

    Parameters:
    -----------
    data : pd.Series
        Numeric variable to analyze
    include_outliers : bool, optional
        Whether to show outliers in the box plot (default: True)
    """
    # Remove NaN values for analysis
    clean_data = data.dropna()

    if len(clean_data) == 0:
        print("⚠️  WARNING: No data available after removing NaN values.")
        return

    var_name = data.name if data.name else "Variable"

    # Calculate statistics
    print("=" * 70)
    print(f"DESCRIPTIVE STATISTICS: {var_name}")
    print("=" * 70)
    print(f"\nCentral Tendency:")
    print(f"  Mean:           {clean_data.mean():.2f}")
    print(f"  Median:         {clean_data.median():.2f}")
    print(
        f"  Mode:           {clean_data.mode().values[0] if len(clean_data.mode()) > 0 else 'N/A'}"
    )

    print(f"\nDispersion:")
    print(f"  Std Dev:        {clean_data.std():.2f}")
    print(f"  Variance:       {clean_data.var():.2f}")
    print(f"  Range:          {clean_data.max() - clean_data.min():.2f}")
    print(
        f"  IQR:            {clean_data.quantile(0.75) - clean_data.quantile(0.25):.2f}"
    )

    print(f"\nQuartiles:")
    print(f"  Min (0%):       {clean_data.min():.2f}")
    print(f"  Q1 (25%):       {clean_data.quantile(0.25):.2f}")
    print(f"  Q2 (50%):       {clean_data.quantile(0.50):.2f}")
    print(f"  Q3 (75%):       {clean_data.quantile(0.75):.2f}")
    print(f"  Max (100%):     {clean_data.max():.2f}")

    print(f"\nShape:")
    print(f"  Skewness:       {clean_data.skew():.3f}")
    print(f"  Kurtosis:       {clean_data.kurtosis():.3f}")

    print(f"\nSample Size:")
    print(f"  Valid:          {len(clean_data)}")
    print(f"  Missing:        {data.isna().sum()}")
    print(f"  Total:          {len(data)}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    axes[0, 0].hist(clean_data, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 0].axvline(
        clean_data.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {clean_data.mean():.2f}",
    )
    axes[0, 0].axvline(
        clean_data.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {clean_data.median():.2f}",
    )
    axes[0, 0].set_xlabel(var_name)
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title(f"Histogram of {var_name}")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Box plot
    bp = axes[0, 1].boxplot(
        clean_data,
        vert=True,
        patch_artist=True,
        showfliers=include_outliers,
        boxprops=dict(facecolor="lightgreen", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="red", markersize=5, alpha=0.5),
    )
    axes[0, 1].set_ylabel(var_name)
    axes[0, 1].set_title(f"Box Plot of {var_name}")
    axes[0, 1].grid(alpha=0.3)

    # KDE plot
    clean_data.plot.kde(ax=axes[1, 0], color="purple", linewidth=2)
    axes[1, 0].axvline(
        clean_data.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {clean_data.mean():.2f}",
    )
    axes[1, 0].axvline(
        clean_data.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {clean_data.median():.2f}",
    )
    axes[1, 0].set_xlabel(var_name)
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title(f"Kernel Density Estimate of {var_name}")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Q-Q plot
    stats.probplot(clean_data, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f"Q-Q Plot of {var_name}")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Additional CDF plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort data for CDF
    sorted_data = np.sort(clean_data)
    # Calculate cumulative probabilities
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Plot CDF
    ax.plot(sorted_data, y, linewidth=2, color="darkblue")
    ax.axhline(
        0.5,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Median (50th percentile)",
    )
    ax.axhline(
        0.25,
        color="orange",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Q1 (25th percentile)",
    )
    ax.axhline(
        0.75,
        color="orange",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Q3 (75th percentile)",
    )
    ax.set_xlabel(var_name)
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"Cumulative Distribution Function (CDF) of {var_name}")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def analyze_categorical_variable(data: pd.Series) -> None:
    """
    Analyze a categorical variable with frequency statistics and visualizations.

    Parameters:
    -----------
    data : pd.Series
        Categorical variable to analyze (object or category dtype)
    """

    var_name = data.name if data.name else "Variable"

    # Calculate frequencies
    value_counts = data.value_counts()
    value_percentages = data.value_counts(normalize=True) * 100

    # Combine counts and percentages
    freq_table = pd.DataFrame({"Count": value_counts, "Percentage": value_percentages})

    # Identify most common and rare categories
    most_common = value_counts.index[0]
    rare_categories = value_percentages[value_percentages < 5].index.tolist()

    # Print summary
    print("=" * 70)
    print(f"CATEGORICAL VARIABLE ANALYSIS: {var_name}")
    print("=" * 70)
    print(f"\nCardinality: {data.nunique()} unique categories")
    print(
        f"Missing values: {data.isna().sum()} ({data.isna().sum() / len(data) * 100:.2f}%)"
    )
    print(f"Total observations: {len(data)}")

    print(
        f"\nMost Common Category: '{most_common}' ({value_counts[most_common]} occurrences, {value_percentages[most_common]:.2f}%)"
    )

    if rare_categories:
        print(f"\nRare Categories (<5%): {len(rare_categories)}")
        for cat in rare_categories:
            print(f"  - '{cat}': {value_counts[cat]} ({value_percentages[cat]:.2f}%)")
    else:
        print("\nNo rare categories found (<5% threshold)")

    print(f"\n{'-' * 70}")
    print("FREQUENCY TABLE:")
    print(f"{'-' * 70}")
    print(freq_table.to_string())
    print("=" * 70)

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    value_counts.plot.bar(ax=axes[0], color="skyblue", edgecolor="black")
    axes[0].set_xlabel(var_name, fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(f"Bar Chart of {var_name}", fontsize=14, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", alpha=0.3)

    # Add count labels on bars
    for i, (idx, val) in enumerate(value_counts.items()):
        axes[0].text(i, val, str(val), ha="center", va="bottom", fontsize=9)

    # Treemap
    colors = plt.cm.Set3(range(len(value_counts)))
    labels = [
        f"{cat}\n{count}\n({pct:.1f}%)"
        for cat, count, pct in zip(
            value_counts.index, value_counts.values, value_percentages.values
        )
    ]

    squarify.plot(
        sizes=value_counts.values,
        label=labels,
        color=colors,
        alpha=0.8,
        ax=axes[1],
        text_kwargs={"fontsize": 10, "weight": "bold"},
    )

    axes[1].set_title(f"Treemap of {var_name}", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_scatter(x_data, y_data, hue=None, alpha=0.6, figsize=(10, 6)):
    """
    Create a scatter plot for two numeric variables.

    Parameters:
    -----------
    x_data : pd.Series
        Data for x-axis
    y_data : pd.Series
        Data for y-axis
    hue : pd.Series, optional
        Categorical variable for color coding points
    alpha : float, default=0.6
        Transparency of points
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    """
    # Get variable names
    x_name = x_data.name if x_data.name else "X Variable"
    y_name = y_data.name if y_data.name else "Y Variable"

    # Combine data and remove NaN values
    if hue is not None:
        hue_name = hue.name if hue.name else "Category"
        combined_df = pd.DataFrame({x_name: x_data, y_name: y_data, hue_name: hue})
    else:
        combined_df = pd.DataFrame({x_name: x_data, y_name: y_data})

    combined_df = combined_df.dropna()

    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize)

    if hue is not None:
        # Get unique categories and create color map
        categories = combined_df[hue_name].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

        # Plot each category separately
        for i, category in enumerate(categories):
            mask = combined_df[hue_name] == category
            ax.scatter(
                combined_df.loc[mask, x_name],
                combined_df.loc[mask, y_name],
                alpha=alpha,
                s=50,
                color=colors[i],
                edgecolors="black",
                linewidth=0.5,
                label=str(category),
            )
    else:
        ax.scatter(
            combined_df[x_name],
            combined_df[y_name],
            alpha=alpha,
            s=50,
            color="steelblue",
            edgecolors="black",
            linewidth=0.5,
        )

    # Add regression line
    z = np.polyfit(combined_df[x_name], combined_df[y_name], 1)
    p = np.poly1d(z)
    ax.plot(
        combined_df[x_name],
        p(combined_df[x_name]),
        "r--",
        linewidth=2,
        label=f"Trend line: y={z[0]:.2f}x+{z[1]:.2f}",
    )

    # Calculate correlation
    correlation = combined_df[x_name].corr(combined_df[y_name])

    ax.set_xlabel(x_name, fontsize=12)
    ax.set_ylabel(y_name, fontsize=12)

    title = f"Scatter Plot: {x_name} vs {y_name}\nCorrelation: {correlation:.3f}"
    if hue is not None:
        title += f" (colored by {hue_name})"
    ax.set_title(title, fontsize=14)

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print correlation info
    print(f"Pearson Correlation: {correlation:.4f}")
    print(f"Number of observations: {len(combined_df)}")


def quick_correlation_matrix(
    df: pd.DataFrame, method: str = "pearson", min_periods: int = 30
) -> pd.DataFrame:
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
    p_matrix = pd.DataFrame(
        np.zeros_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns
    )

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            data1 = df[col1].dropna()
            data2 = df[col2].dropna()

            # Find common indices
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) >= min_periods:
                if method == "pearson":
                    _, p_val = pearsonr(data1[common_idx], data2[common_idx])
                else:
                    _, p_val = spearmanr(data1[common_idx], data2[common_idx])

                p_matrix.loc[col1, col2] = p_val
                p_matrix.loc[col2, col1] = p_val

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=axes[0],
        cbar_kws={"label": f"{method.capitalize()} Correlation"},
    )
    axes[0].set_title(f"{method.capitalize()} Correlation Matrix")

    # Significance heatmap
    sig_matrix = p_matrix < 0.05
    sns.heatmap(
        sig_matrix,
        mask=mask,
        cmap="RdYlGn",
        center=0.5,
        ax=axes[1],
        cbar_kws={"label": "Significant (p < 0.05)"},
    )
    axes[1].set_title("Statistical Significance")

    plt.tight_layout()
    plt.show()

    return corr_matrix


def analyze_categorical_categorical(cat_data1, cat_data2, alpha=0.05):
    """
    Analyze relationship between two categorical variables using chi-square test.

    Parameters:
    -----------
    cat_data1 : pd.Series
        First categorical variable
    cat_data2 : pd.Series
        Second categorical variable
    alpha : float, default=0.05
        Significance level

    Returns:
    --------
    dict : Dictionary containing test results
    """

    # Get column names
    col1 = cat_data1.name if cat_data1.name else "Variable 1"
    col2 = cat_data2.name if cat_data2.name else "Variable 2"

    # Combine data and remove NaN
    combined_df = pd.DataFrame({col1: cat_data1, col2: cat_data2})
    combined_df = combined_df.dropna()

    # Create contingency table
    contingency = pd.crosstab(combined_df[col1], combined_df[col2])

    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Calculate Cramér's V for effect size
    n = contingency.sum().sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim))

    # Interpret effect size
    if cramers_v < 0.1:
        effect_interpretation = "Negligible association"
    elif cramers_v < 0.3:
        effect_interpretation = "Weak association"
    elif cramers_v < 0.5:
        effect_interpretation = "Moderate association"
    else:
        effect_interpretation = "Strong association"

    # Overall interpretation
    is_significant = p_value < alpha
    if is_significant:
        interpretation = (
            f"SIGNIFICANT association detected (p={p_value:.6f}). "
            f"{col1} and {col2} are related. "
            f"{effect_interpretation} (Cramér's V = {cramers_v:.3f})."
        )
    else:
        interpretation = (
            f"NO significant association (p={p_value:.6f}). "
            f"{col1} and {col2} appear to be independent."
        )

    # Visualizations
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Heatmap of contingency table
    sns.heatmap(
        contingency,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=axes[0],
        cbar_kws={"label": "Count"},
    )
    axes[0].set_title(f"Contingency Table: {col1} vs {col2}")
    axes[0].set_xlabel(col2)
    axes[0].set_ylabel(col1)

    # Stacked bar chart with proportions
    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0)
    ax = contingency_pct.plot(
        kind="bar", stacked=True, ax=axes[1], colormap="viridis", alpha=0.8
    )

    # Add proportion labels on bars
    for container in ax.containers:
        labels = [
            f"{v.get_height():.2%}" if v.get_height() > 0.05 else "" for v in container
        ]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=9)

    axes[1].set_title(f"Proportional Distribution: {col1} by {col2}")
    axes[1].set_xlabel(col1)
    axes[1].set_ylabel("Proportion")
    axes[1].legend(title=col2, bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, ha="center")

    plt.tight_layout()
    plt.show()

    # Print results
    print("=" * 70)
    print(f"CHI-SQUARE TEST OF INDEPENDENCE: {col1} vs {col2}")
    print("=" * 70)

    print(f"\nContingency Table:")
    print(contingency)

    print(f"\nTest Statistics:")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  Degrees of freedom:   {dof}")
    print(f"  P-value:              {p_value:.6f}")
    print(f"  Cramér's V:           {cramers_v:.3f}")
    print(f"  Effect size:          {effect_interpretation}")
    print(f"  Significant at α={alpha}: {'YES' if is_significant else 'NO'}")

    print(f"\nInterpretation:")
    print(f"  {interpretation}")

    # Check for cells with low expected frequencies
    low_expected = (expected < 5).sum()
    if low_expected > 0:
        print(f"\n⚠️  WARNING: {low_expected} cell(s) have expected frequency < 5.")
        print(f"   Chi-square test may not be reliable. Consider Fisher's exact test.")


def analyze_categorical_numerical(
    cat_data, num_data, alpha=0.05, include_outliers=True
):
    """
    Analyze relationship between a categorical and numerical variable.

    Parameters:
    -----------
    cat_data : pd.Series
        Categorical variable (can be encoded as numerical)
    num_data : pd.Series
        Numerical variable
    alpha : float, default=0.05
        Significance level
    include_outliers : bool, optional
        Whether to show outliers in the box plot (default: True)

    Returns:
    --------
    dict : Dictionary containing test results
    """

    # Get variable names
    cat_name = cat_data.name if cat_data.name else "Categorical Variable"
    num_name = num_data.name if num_data.name else "Numerical Variable"

    # Combine data and remove NaN values
    combined_df = pd.DataFrame({cat_name: cat_data, num_name: num_data})
    combined_df = combined_df.dropna()

    # Convert categorical to string to ensure proper grouping
    combined_df[cat_name] = combined_df[cat_name].astype(str)

    # Get groups
    groups = combined_df[cat_name].unique()
    n_groups = len(groups)

    if n_groups < 2:
        print(
            f"⚠️  ERROR: Need at least 2 groups for analysis. Found {n_groups} group(s)."
        )
        return None

    # Create group data
    group_data = [
        combined_df[combined_df[cat_name] == group][num_name].values for group in groups
    ]

    # Perform statistical test
    if n_groups == 2:
        # T-test for 2 groups
        stat, p_value = ttest_ind(group_data[0], group_data[1], equal_var=False)
        test_name = "Welch's t-test"

        # Cohen's d effect size
        mean1, mean2 = np.mean(group_data[0]), np.mean(group_data[1])
        std1, std2 = np.std(group_data[0], ddof=1), np.std(group_data[1], ddof=1)
        n1, n2 = len(group_data[0]), len(group_data[1])
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std
        effect_size = abs(cohens_d)
        effect_measure = "Cohen's d"

        # Interpret Cohen's d
        if effect_size < 0.2:
            effect_interpretation = "Negligible effect"
        elif effect_size < 0.5:
            effect_interpretation = "Small effect"
        elif effect_size < 0.8:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"

    else:
        # ANOVA for 3+ groups
        stat, p_value = f_oneway(*group_data)
        test_name = "One-Way ANOVA"

        # Eta-squared effect size
        grand_mean = combined_df[num_name].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
        ss_total = sum((combined_df[num_name] - grand_mean) ** 2)
        eta_squared = ss_between / ss_total
        effect_size = eta_squared
        effect_measure = "Eta-squared (η²)"

        # Interpret eta-squared
        if effect_size < 0.01:
            effect_interpretation = "Negligible effect"
        elif effect_size < 0.06:
            effect_interpretation = "Small effect"
        elif effect_size < 0.14:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"

    # Overall interpretation
    is_significant = p_value < alpha
    if is_significant:
        interpretation = (
            f"SIGNIFICANT difference detected (p={p_value:.6f}). "
            f"{cat_name} has a significant effect on {num_name}. "
            f"{effect_interpretation} ({effect_measure} = {effect_size:.3f})."
        )
    else:
        interpretation = (
            f"NO significant difference (p={p_value:.6f}). "
            f"{cat_name} does not significantly affect {num_name}."
        )

    # Create visualizations
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Box plot
    combined_df.boxplot(
        column=num_name,
        by=cat_name,
        ax=axes[0],
        patch_artist=True,
        grid=False,
        showfliers=include_outliers,
    )
    axes[0].set_title(f"Box Plot: {num_name} by {cat_name}")
    axes[0].set_xlabel(cat_name)
    axes[0].set_ylabel(num_name)
    plt.sca(axes[0])
    plt.xticks(rotation=0, ha="center")

    # Violin plot
    sns.violinplot(
        data=combined_df,
        x=cat_name,
        y=num_name,
        hue=cat_name,
        ax=axes[1],
        palette="Set2",
        inner="box",
        legend=False,
    )
    axes[1].set_title(f"Violin Plot: {num_name} by {cat_name}")
    axes[1].set_xlabel(cat_name)
    axes[1].set_ylabel(num_name)
    axes[1].tick_params(axis="x", rotation=0)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0, ha="center")

    plt.tight_layout()
    plt.show()

    # Print results
    print("=" * 70)
    print(f"{test_name.upper()}: {num_name} by {cat_name}")
    print("=" * 70)

    print(f"\nDescriptive Statistics by Group:")
    for group in groups:
        group_values = combined_df[combined_df[cat_name] == group][num_name]
        print(f"\n  {cat_name} = {group}:")
        print(f"    N:      {len(group_values)}")
        print(f"    Mean:   {group_values.mean():.4f}")
        print(f"    Median: {group_values.median():.4f}")
        print(f"    Std:    {group_values.std():.4f}")
        print(f"    Min:    {group_values.min():.4f}")
        print(f"    Max:    {group_values.max():.4f}")

    print(f"\nTest Statistics:")
    print(f"  Test:                {test_name}")
    print(f"  Test statistic:      {stat:.4f}")
    print(f"  P-value:             {p_value:.6f}")
    print(f"  {effect_measure}:    {effect_size:.3f}")
    print(f"  Effect size:         {effect_interpretation}")
    print(f"  Significant at α={alpha}: {'YES' if is_significant else 'NO'}")

    print(f"\nInterpretation:")
    print(f"  {interpretation}")

    # Check assumptions
    print(f"\nAssumption Checks:")

    # Check normality for each group (Shapiro-Wilk test)
    print(f"  Normality (Shapiro-Wilk test):")
    for group in groups:
        group_values = combined_df[combined_df[cat_name] == group][num_name]
        if len(group_values) >= 3:
            _, p_norm = stats.shapiro(group_values)
            normality_status = "Normal" if p_norm > 0.05 else "Non-normal"
            print(f"    {cat_name} = {group}: p={p_norm:.4f} ({normality_status})")

    # Check homogeneity of variance (Levene's test)
    _, p_levene = stats.levene(*group_data)
    variance_status = "Equal variances" if p_levene > 0.05 else "Unequal variances"
    print(
        f"  Homogeneity of variance (Levene's test): p={p_levene:.4f} ({variance_status})"
    )

    if p_levene < 0.05 and n_groups == 2:
        print(f"   ⚠️  NOTE: Welch's t-test was used (does not assume equal variances)")


def analyze_numerical_numerical(x_data, y_data, alpha=0.05):
    """
    Comprehensive analysis of relationship between two numerical variables.

    Parameters:
    -----------
    x_data : pd.Series
        First numerical variable (independent)
    y_data : pd.Series
        Second numerical variable (dependent)
    alpha : float, default=0.05
        Significance level for hypothesis tests
    """

    # Get variable names
    x_name = x_data.name if x_data.name else "X Variable"
    y_name = y_data.name if y_data.name else "Y Variable"

    # Combine data and remove NaN values
    combined_df = pd.DataFrame({x_name: x_data, y_name: y_data})
    combined_df = combined_df.dropna()

    if len(combined_df) < 3:
        print(f"⚠️  ERROR: Need at least 3 observations. Found {len(combined_df)}.")
        return None

    x_clean = combined_df[x_name].values
    y_clean = combined_df[y_name].values

    # Calculate correlation coefficients
    pearson_r, pearson_p = pearsonr(x_clean, y_clean)
    spearman_rho, spearman_p = spearmanr(x_clean, y_clean)

    # Fit linear regression
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)
    y_pred = p(x_clean)

    # Calculate R-squared
    r_squared = r2_score(y_clean, y_pred)

    # Interpret correlation strength
    abs_pearson = abs(pearson_r)
    if abs_pearson < 0.1:
        strength = "Negligible"
    elif abs_pearson < 0.3:
        strength = "Weak"
    elif abs_pearson < 0.5:
        strength = "Moderate"
    elif abs_pearson < 0.7:
        strength = "Strong"
    else:
        strength = "Very Strong"

    direction = "positive" if pearson_r > 0 else "negative"

    # Print results
    print("=" * 70)
    print(f"BIVARIATE CORRELATION ANALYSIS: {x_name} vs {y_name}")
    print("=" * 70)

    print(f"\nSample Size:")
    print(f"  Valid observations: {len(combined_df)}")
    print(f"  Missing values:     {len(x_data) - len(combined_df)}")

    print(f"\nCorrelation Coefficients:")
    print(f"  Pearson's r:        {pearson_r:.4f}")
    print(f"    P-value:          {pearson_p:.6f}")
    print(f"    Significant:      {'YES' if pearson_p < alpha else 'NO'} (α={alpha})")
    print(f"  Spearman's ρ:       {spearman_rho:.4f}")
    print(f"    P-value:          {spearman_p:.6f}")
    print(f"    Significant:      {'YES' if spearman_p < alpha else 'NO'} (α={alpha})")

    print(f"\nRegression Statistics:")
    print(f"  R-squared (R²):     {r_squared:.4f}")
    print(f"  Slope:              {z[0]:.4f}")
    print(f"  Intercept:          {z[1]:.4f}")
    print(f"  Equation:           y = {z[0]:.4f}x + {z[1]:.4f}")

    print(f"\nInterpretation:")
    print(
        f"  {strength} {direction} linear relationship between {x_name} and {y_name}."
    )
    print(f"  {r_squared*100:.2f}% of variance in {y_name} is explained by {x_name}.")

    # Create visualizations
    fig = plt.figure(figsize=(16, 6))

    # Scatter plot with regression line
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(
        x_clean,
        y_clean,
        alpha=0.6,
        s=50,
        color="steelblue",
        edgecolors="black",
        linewidth=0.5,
    )
    ax1.plot(x_clean, y_pred, "r--", linewidth=2, label=f"y = {z[0]:.4f}x + {z[1]:.4f}")
    ax1.set_xlabel(x_name, fontsize=12)
    ax1.set_ylabel(y_name, fontsize=12)
    ax1.set_title(
        f"Scatter Plot with Regression Line\nPearson r = {pearson_r:.3f}, R² = {r_squared:.3f}",
        fontsize=13,
    )
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Contour plot (2D density)
    ax2 = plt.subplot(1, 2, 2)

    # Create a grid for contour plot
    x_min, x_max = x_clean.min(), x_clean.max()
    y_min, y_max = y_clean.min(), y_clean.max()

    # Add margin
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    xx, yy = np.mgrid[
        x_min - x_margin : x_max + x_margin : 100j,
        y_min - y_margin : y_max + y_margin : 100j,
    ]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x_clean, y_clean])

    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Plot contours
    contour = ax2.contourf(xx, yy, f, levels=15, cmap="YlOrRd", alpha=0.7)
    ax2.scatter(
        x_clean,
        y_clean,
        alpha=0.4,
        s=30,
        color="black",
        edgecolors="white",
        linewidth=0.5,
    )
    ax2.plot(
        x_clean, y_pred, "blue", linewidth=2, linestyle="--", label="Regression line"
    )

    plt.colorbar(contour, ax=ax2, label="Density")
    ax2.set_xlabel(x_name, fontsize=12)
    ax2.set_ylabel(y_name, fontsize=12)
    ax2.set_title(
        f"2D Density Contour Plot\nSpearman ρ = {spearman_rho:.3f}", fontsize=13
    )
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "r_squared": r_squared,
        "slope": z[0],
        "intercept": z[1],
        "n_observations": len(combined_df),
        "strength": strength,
        "direction": direction,
        "x_variable": x_name,
        "y_variable": y_name,
    }
