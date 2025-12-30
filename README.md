# Thessaloniki Short-Term Rental Market Analysis

A data-driven examination of 4,124 Airbnb listings to inform sustainable tourism policy.

![Dashboard Overview](images/Dashboard%20overview.png)

---

## Overview

Thessaloniki's short-term rental market has experienced explosive growth in recent years. This project analyzes regulatory compliance, host ecosystem dynamics, geographic performance patterns, and temporal quality trends to answer one question: **Is the market healthy?**

**Short answer:** Mostly yes, but with caveats worth monitoring.

The analysis reveals a competitive market where scale provides no revenue advantage, but quality systematically declines with host size. Mid-scale operators (2-10 listings) emerge as the market's quality backbone, yet they're being squeezed by polarization toward individual hosts and large commercial operators.

---

## Key Findings

| Dimension | Finding |
|-----------|---------|
| **Compliance** | 97.3% licensing rate; 2 hosts control 42% of exemptions |
| **Host Quality** | 0.21-star gap between Individual (4.92★) and Large Multi (4.71★) hosts |
| **Superhost Premium** | 3.2x revenue multiplier for individuals vs 1.6x for large operators |
| **Geography** | 86% of listings within 3km of center; downtown competition elevates quality (+13% superhost rate for Large Multi) |
| **Trajectory** | Large Multi hosts nearly doubled market share post-pandemic (25% → 43%) |
| **Quality Variance** | New listings show 2.4x rating variance vs established ones |

![Review Scores by Host Type](images/Review%20scores%20by%20host%20types.png)

---

## Project Structure

```
airbnb-rental-market-analysis-thessaloniki/
├── data/
│   ├── raw/                        # Original Inside Airbnb data
│   └── processed/                  # Cleaned analysis-ready datasets
├── images/                         # Exported visualizations
├── notebooks/
│   ├── regulatory_compliance.ipynb
│   ├── host_type_impact.ipynb
│   ├── geographic_performance.ipynb
│   └── temporal_dynamics.ipynb
├── powerbi/
│   └── DAX_formulas.md
├── scripts/
│   ├── data_preprocessing.py
│   ├── calendar_preprocessing.py
│   ├── data_preprocessing_log.md
│   └── eda_functions.py
├── report.md                       # Full analytical report
├── requirements.txt
└── README.md
```

---

## Data Validation

Raw data from Inside Airbnb underwent validation and cleaning before analysis:

- **Anonymization**: Host IDs, listing IDs, and licenses hashed; coordinates rounded to ~11m accuracy
- **Outlier removal**: Extreme `minimum_nights` and price outliers excluded
- **Inactive filtering**: Dead listings (zero activity + missing reviews) removed
- **Missing value imputation**: Host categories imputed from actual listing counts per host
- **Feature engineering**: Distance zones, price segments, market maturity categories

See [`scripts/data_preprocessing_log.md`](scripts/data_preprocessing_log.md) for full methodology.

**Final sample**: 4,124 licensed listings after compliance and data validation filtering.

---

## Explore the Analysis

| Notebook | Description |
|----------|-------------|
| [Regulatory Compliance](notebooks/regulatory_compliance.ipynb) | License distribution, exemption concentration, compliance anomalies |
| [Host Type Impact](notebooks/host_type_impact.ipynb) | Scale vs quality dynamics, superhost achievement, revenue analysis |
| [Geographic Performance](notebooks/geographic_performance.ipynb) | Spatial clustering, downtown quality paradox, zone-based pricing |
| [Temporal Dynamics](notebooks/temporal_dynamics.ipynb) | Market maturity trends, quality divergence, Large multihosts trajectory |

---

## Methodology

**Data Source:** Inside Airbnb (June 2025 snapshot)

**Statistical Approach:**
- Non-parametric tests (Kruskal-Wallis, Mann-Whitney U) for group comparisons
- Effect size reporting (ε², Cramér's V, rank-biserial correlation) alongside p-values
- Levene's test for variance homogeneity

**Geographic Reference:** White Tower / Aristotelous Square midpoint (40.62962°N, 22.94473°E)

**Host Categorization:**
| Category | Listings | Profile |
|----------|----------|---------|
| Individual | 1 | Casual hosts |
| Small Multi | 2-3 | Semi-professional |
| Medium Multi | 4-10 | Professional operators |
| Large Multi | 11+ | Commercial operations |

---

## Reports & Dashboards

| Resource | Description |
|----------|-------------|
| [Full Report](report.md) | Narrative analysis with statistical findings and policy implications |
| [Power BI Dashboard](powerbi/) | Interactive exploration of key metrics |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/ntinasf/airbnb-rental-market-analysis-thessaloniki.git
cd airbnb-rental-market-analysis-thessaloniki

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/
```

---

## Data Sources

| Dataset | Source | Records |
|---------|--------|---------|
| Listings | [Inside Airbnb](http://insideairbnb.com/) | 4,700+ raw → 4,124 cleaned |
| Calendar | Inside Airbnb | ~1.7M rows |
| Neighbourhoods | Inside Airbnb | GeoJSON |

---

## Author

**[Placeholder: Your Name]**

[Portfolio](https://placeholder-portfolio.com) | [LinkedIn](https://linkedin.com/in/placeholder)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*This project was developed as a data analytics portfolio demonstration.*