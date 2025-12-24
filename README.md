# Thessaloniki Short-Term Rental Market Analysis

A data-driven examination of 4,124 Airbnb listings to inform sustainable tourism policy.

![Placeholder: Dashboard Overview](pictures/dashboard_overview.png)

---

## Overview

Thessaloniki's short-term rental market has grown rapidly in recent years. This project analyzes regulatory compliance, host ecosystem dynamics, geographic performance patterns, and temporal quality trends to provide evidence-based policy recommendations.

**Key question:** Is the market healthy, and where should policymakers focus attention?

**Short answer:** Mostly yes, with caveats worth monitoring.

---

## Key Findings

| Dimension | Finding | Effect Size |
|-----------|---------|-------------|
| Compliance | 97.3% licensing rate; 2 hosts control 42% of exemptions | - |
| Host Quality | 0.27-star gap between Individual (4.90) and Large Multi (4.70) hosts | ε² = 0.12 |
| Superhost Premium | 3.2x revenue multiplier for individuals vs 1.7x for large operators | Large effect |
| Geography | 86% of listings within 3km of center; downtown elevates quality | ε² = 0.19 (location ratings) |
| Trajectory | Large Multi hosts doubled market share post-pandemic (25% to 43%) | - |
| Quality Variance | New listings show 2.4x rating variance vs established ones | p < 0.001 |

Mid-scale operators (2-10 listings) emerge as the market's quality backbone with highest superhost rates (45-47%).

---

## Project Structure

```
airbnb-rental-market-analysis-thessaloniki/
│
├── data/
│   ├── raw/                      # Original Inside Airbnb data
│   └── processed/                # Cleaned analysis-ready datasets
│
├── notebooks/
│   ├── regulatory_compliance.ipynb
│   ├── host_type_impact.ipynb
│   ├── geographic_performance.ipynb
│   └── temporal_dynamics.ipynb
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── calendar_preprocessing.py
│   └── eda_functions.py
│
├── powerbi/
│   └── DAX_formulas.md
│
├── pictures/                     # Visualizations for README and reports
│
├── report.md                     # Full analytical report
├── executive_summary.md          # Policy brief
├── requirements.txt
└── README.md
```

---

## Methodology

**Data Source:** Inside Airbnb (June 2025 snapshot)

**Sample:** 4,124 licensed listings after compliance filtering

**Statistical Approach:**
- Non-parametric tests (Kruskal-Wallis, Mann-Whitney U) for group comparisons
- Effect size reporting (ε², Cramer's V, rank-biserial correlation)
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

## Visualizations

### Host Ecosystem
![Placeholder: Review Scores by Host Type](pictures/review_scores_host_type.png)

### Geographic Patterns
![Placeholder: Superhost Rate by Zone](pictures/superhost_by_zone.png)

### Temporal Dynamics
![Placeholder: Quality Variance by Market Maturity](pictures/quality_variance_maturity.png)

---

## Reports

| Document | Description |
|----------|-------------|
| [Full Report](report.md) | Detailed analysis with statistical findings |
| [Executive Summary](executive_summary.md) | Policy brief with recommendations |
| [Power BI Dashboard](powerbi/) | Interactive exploration |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/[username]/airbnb-rental-market-analysis-thessaloniki.git
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
| Listings | [Inside Airbnb](http://insideairbnb.com/) | 4,700+ |
| Calendar | Inside Airbnb | ~1.7M rows |
| Neighbourhoods | Inside Airbnb | GeoJSON |

All data publicly available. Analysis conducted on June 2025 snapshot.

---

## Author

**[Your Name]**

[Portfolio](https://yourwebsite.com) | [LinkedIn](https://linkedin.com/in/yourprofile)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*This project was developed as a data analytics portfolio demonstration.*