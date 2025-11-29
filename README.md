# Thessaloniki Airbnb Market Analysis
### Evidence-Based Policy Recommendations for Sustainable Tourism Growth

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow.svg)](link-to-dashboard)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

---

## 📋 Executive Summary

This project analyzes Thessaloniki's short-term rental market (4,817 Airbnb listings) to provide data-driven policy recommendations for the Tourism Development Authority. Through rigorous statistical analysis, we identified critical patterns in regulatory compliance, host ecosystem dynamics, and market quality trends.

### Key Findings

| Finding | Metric | Policy Implication |
|---------|--------|-------------------|
| **License Concentration Risk** | Single host controls 29.3% of exemptions | Reform exemption system |
| **Superhost Quality Premium** | 3.8x revenue advantage for Individual hosts | Incentivize quality certification |
| **Geographic Stratification** | 60% commercial operators downtown vs 27% in neighborhoods | Preserve neighborhood diversity |
| **Budget Segment Crisis** | -3.84% quality decline, Large Multi share 43%→62% | Urgent quality standards needed |

---

## 🎯 Business Problem

**Primary Research Question:**
> How can Thessaloniki optimize its short-term rental ecosystem to maximize tourism benefits while ensuring neighborhood sustainability and regulatory compliance?

**Stakeholders:** Tourism Development Authority, City Council, Host Community, Residents

**Analysis Scope:** 4,379 regular licensed listings (March 2025 data)

---

## 📊 Methodology

### Analytical Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH DESIGN                          │
├─────────────────────────────────────────────────────────────┤
│  H1: Regulatory Compliance                                  │
│      └─ License concentration & duplicate patterns          │
│                                                             │
│  H3: Host Ecosystem Dynamics                                │
│      └─ Quality premium analysis (Superhost × Host Type)    │
│                                                             │
│  H5: Geographic & Temporal Stratification                   │
│      └─ Spatial patterns + Quality evolution trends         │
└─────────────────────────────────────────────────────────────┘
```

### Statistical Methods

| Hypothesis | Method | Key Statistic |
|------------|--------|---------------|
| H1 | Descriptive Analysis, Concentration Metrics | Top-host share: 29.3% |
| H3 | Two-way ANOVA, Post-hoc Tukey HSD | F(5,4373) = 89.2, p < 0.001 |
| H5 | Chi-square, Cramér's V, Temporal Regression | χ²(6) = 89.1, V = 0.142 |

### Tools & Technologies

- **Data Processing:** Python 3.9+ (pandas, numpy)
- **Statistical Analysis:** scipy, statsmodels
- **Visualization:** matplotlib, seaborn, plotly
- **Dashboard:** Microsoft Power BI
- **Version Control:** Git/GitHub

---

## 📁 Repository Structure

```
thessaloniki-airbnb-analysis/
│
├── 📂 data/
│   ├── raw/                    # Original datasets (not tracked)
│   ├── processed/              # Cleaned analysis-ready data
│   └── data_dictionary.md      # Variable definitions
│
├── 📂 notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   └── 04_insights_synthesis.ipynb
│
├── 📂 src/
│   ├── data_processing.py      # Data cleaning functions
│   ├── analysis_utils.py       # Statistical test helpers
│   └── visualization.py        # Custom plotting functions
│
├── 📂 reports/
│   ├── executive_summary.pdf
│   ├── technical_report.pdf
│   └── presentation_slides.pdf
│
├── 📂 dashboard/
│   ├── thessaloniki_airbnb.pbix
│   └── screenshots/
│
├── 📂 docs/
│   ├── methodology.md
│   └── data_sources.md
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Power BI Desktop (for dashboard)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/thessaloniki-airbnb-analysis.git
cd thessaloniki-airbnb-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Analysis

```bash
# Run notebooks in sequence
jupyter notebook notebooks/01_data_preparation.ipynb
# Continue with 02, 03, 04...

# Or run all preprocessing
python src/data_processing.py
```

---

## 📈 Key Visualizations

### Dashboard Overview
![Dashboard Overview](dashboard/screenshots/overview.png)

### Superhost Premium Analysis
![Superhost Premium](dashboard/screenshots/host_ecosystem.png)

### Quality Decline Trend
![Quality Decline](dashboard/screenshots/market_evolution.png)

*Full interactive dashboard available in Power BI file*

---

## 📊 Data Sources

| Dataset | Source | Records | Date |
|---------|--------|---------|------|
| Listings | [Inside Airbnb](http://insideairbnb.com/) | 4,817 | March 2025 |
| Calendar | Inside Airbnb | ~1.7M | March 2025 |
| Reviews | Inside Airbnb | ~150K | March 2025 |

**Data Ethics:** All data is publicly available. Personal information has been anonymized for this analysis.

---

## 📋 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
statsmodels>=0.13.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0
jupyter>=1.0.0
```

---

## 🔍 Detailed Findings

### H1: Regulatory Compliance

**Finding:** High overall compliance (90.9%) masks concentration risk in exemptions.

- Single host controls 29.3% of all license exemptions (12 of 41 properties)
- 411 listings operate under duplicate license arrangements
- Geographic clustering suggests coordinated regulatory arbitrage

**Recommendation:** Limit exemptions to 2 properties per host; audit duplicate licenses.

### H3: Host Ecosystem Dynamics

**Finding:** Quality certification (Superhost) yields dramatically different returns by operator type.

- Individual hosts: 3.8x revenue premium with Superhost status
- Large Multi operators: Only 1.4x premium
- "Sweet spot" at 2-3 properties: 42% Superhost rate (highest)

**Recommendation:** Incentivize Superhost attainment, especially for multi-property operators.

### H5: Geographic & Temporal Patterns

**Finding:** Market commercialization correlates with quality degradation.

- Downtown: 60% multi-property operators (highest concentration)
- Budget segment: Large Multi share grew 43%→62% while ratings fell 3.84%
- Causal pattern: Commercialization without quality oversight = market failure

**Recommendation:** Implement geographic diversity requirements; urgent quality standards for budget segment.

---

## 📄 Deliverables

| Deliverable | Description | Link |
|-------------|-------------|------|
| **Power BI Dashboard** | Interactive 5-page analytics dashboard | [View](dashboard/) |
| **Executive Summary** | 4-page policy brief for Tourism Authority | [PDF](reports/executive_summary.pdf) |
| **Technical Report** | Full methodology and statistical analysis | [PDF](reports/technical_report.pdf) |
| **Presentation** | Stakeholder presentation slides | [PDF](reports/presentation_slides.pdf) |

---

## 🛠️ Future Work

- [ ] Incorporate review text sentiment analysis
- [ ] Build predictive model for listing success factors
- [ ] Expand to comparative analysis with similar Greek cities
- [ ] Develop automated monitoring dashboard for ongoing tracking

---

## 👤 Author

**[Your Name]**

- Portfolio: [yourwebsite.com](https://yourwebsite.com)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Inside Airbnb](http://insideairbnb.com/) for open data initiative
- Thessaloniki Tourism Development Authority (simulated client)
- [Google's Good Data Analysis Practices](https://research.google/pubs/) for methodology guidance

---

<p align="center">
  <i>This project was completed as a portfolio demonstration of data analytics capabilities.</i>
</p>