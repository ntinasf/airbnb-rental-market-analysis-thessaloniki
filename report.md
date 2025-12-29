# Thessaloniki's Short-Term Rental Market: A Data-Driven Analysis

*An evidence-based examination of 4,124 Airbnb listings to inform sustainable tourism policy*

---

## The Story in Numbers

Thessaloniki is Greece's second city, a historic cultural hub and an increasingly popular destination. During recent years, it has experienced a short-term rental boom. But behind the captivating sunset photos lies a more complex story about who's running the city's tourism infrastructure, where the money flows, and whether rapid growth is coming at the cost of guest experience.

This analysis examines **4,124 licensed Airbnb listings** to answer a simple question: *Is Thessaloniki's STR market healthy?*

The short answer: mostly yes, but with caveats worth noting.
  
---

## Before We Begin: The Compliance Question

Before analyzing market dynamics, we needed to establish a clean dataset. Thessaloniki's regulatory framework requires property registration for short-term rentals (AMA number), with exemptions available for specific property categories.

**The good news**: 97.3% of active listings hold valid licenses.

**The less good news**: The exemption system shows signs of concentration that warrant scrutiny.

![Distribution of License Types](images/Distribution%20of%20license%20types.png)

| License Status | Listings | Share |
|---------------|----------|-------|
| Regular Licensed | 4,124 | 97.3% |
| Exempt | 24 | 0.6% |
| Unlicensed/NA | 17 | 0.4% |
| Suspicious Duplicates | 74 | 1.7% |

Among exempt listings, **two hosts control 42% of all exemptions**—a concentration that raises questions about whether exemption criteria are being applied consistently. Additionally, 6 licenses appear across multiple hosts at different locations, suggesting potential license-sharing irregularities.

![Concentration of Exempt Listings by Host](images/Exempt%20listings%20by%20host.png)

**For auditors**: The 115 non-compliant listings were excluded from further analysis. Of these, only 4 appear actively operational—the rest seem to be deactivated or dormant properties. Focus compliance efforts accordingly.

With the regulatory landscape mapped, we turn to the 4,124 legitimate operators who can paint the real picture of Thessaloniki's tourism accommodation sector.

---

## Who Runs Thessaloniki's Airbnb Market?

The answer is more nuanced than you might expect.

Thessaloniki's STR market operates on two parallel tracks: a fragmented base of smaller hosts (mostly single-listing) and a concentrated tier of commercial operators.

![Distribution of Listings by Host Category](images/Listings%20by%20host%20category.png)

| Host Category | Listings | Market Share | Profile |
|--------------|----------|--------------|---------|
| **Individual** (1 listing) | 1,003 | 24% | Casual hosts, often personal space |
| **Small Multi** (2-3 listings) | 765 | 19% | Semi-professional, growing portfolio |
| **Medium Multi** (4-10 listings) | 780 | 19% | Professional operators |
| **Large Multi** (11+ listings) | 1,576 | 38% | Commercial/corporate operations |

The math is stark: while most hosts are small operators, most listings belong to commercial players. A single operator controls nearly **7% of Thessaloniki's entire Airbnb inventory**. The boxplot reveals extreme outliers—some hosts manage portfolios spanning 200-1,000+ properties globally.

![Distribution of Host Total Listings Count](images/Host%20total%20listing%20count.png)

This isn't inherently problematic. Professional operators can bring efficiency, consistent standards, and tourism infrastructure to markets that need it. The question is whether scale delivers value to guests—or extracts it.

### Bigger ≠ Better

Let's see how small and large operators compare.

**Revenue per listing**? Essentially identical across host categories. Median annual revenue ranges from €1,980 (Medium Multi) to €2,688 (Large Multi), a statistically significant result but practically meaningless (ε² = 0.007, negligible effect size).

![Estimated Revenue Across Host Categories](images/Estimated%20revenue%20across%20host%20types.png)

**12-Month Occupancy**? Same story. The effect size (ε² = 0.006) suggests scale provides zero competitive advantage in filling beds.

![Estimated Occupancy Across Host Categories](images/Estimated%20occupancy%20across%20host%20types.png)

One would expect professional operators to outperform through pricing algorithms, aggressive marketing, or operational efficiency, but they don't. The market appears genuinely competitive as no structural advantage stems from scale alone.

But here's where it gets interesting.

### Where Scale Fails: The Quality Gap

Guest satisfaction tells a different story than revenue metrics.

![Review Scores by Host Type](images/Review%20scores%20by%20host%20types.png)

Review ratings decline systematically as host scale increases:

| Host Category | Median Rating | Variance |
|--------------|---------------|----------|
| Individual | 4.92★ | Lowest |
| Small Multi | 4.90★ | Low |
| Medium Multi | 4.86★ | Moderate |
| Large Multi | 4.71★ | Highest |

This isn't noise. Host category explains **12% of variance in review scores** (ε² = 0.12), a moderate-to-large effect that represents real differences in guest experience. The 0.21 star gap between Individual and Large multihosts may seem small, but in Airbnb's rating ecosystem, it's the difference between "excellent" and "good."

Equally telling is the difference in variance: large operators deliver inconsistent experiences, while smaller hosts cluster tightly around high scores.

---

## The Superhost Factor

Airbnb's Superhost badge is awarded to hosts for sustained excellence in key metrics:response rate (>90%), reviews (>4.8★), cancellations (<1%), and booking volume. It serves as a useful proxy for quality commitment. Who earns it?

![Superhost Achievement by Host Scale](images/Superhost%20achievement%20by%20scale.png)

| Host Category | Superhost Rate |
|--------------|----------------|
| Individual | 39% |
| Small Multi | 47% |
| Medium Multi | 45% |
| Large Multi | 32% |

Here **mid-scale operators (2-10 listings) achieve the highest superhost rates**, suggesting a "growth mindset" where operators invest in quality to build their business. Large operators, managing 11+ properties, appear to prioritize operational efficiency over personalized service.

### The Superhost Premium: Quality Pays (More for Some)

Does the quality investment pay off? Absolutely, but unevenly across categories.

![Superhost Revenue Premium by Host Category](images/Superhost%20premium%20by%20host%20category.png)

| Host Category | Mean Revenue Multiplier (SH vs Non-SH) |
|--------------|-----------------------------------|
| Individual | **3.23×** |
| Small Multi | **3.19×** |
| Medium Multi | 2× |
| Large Multi | 1.6× |

For smaller hosts, Superhost status is a key differentiator. It is a quality badge that enables competition against commercial scale through reputation. For large operators, the quality premium is thinner as guests may already expect professionalism and high quality services from big players.

This us a sign of a troubling dynamic: **quality investment is less rewarded at scale**, potentially incentivizing volume-over-excellence strategies among commercial operators.

### The Quality Floor Problem

The data reveals a concerning fact: **non-Superhost Large operators** deliver the market's worst guest experience at **4.42★**—0.3 stars below the market average.

![Quality Gap - Superhost vs Non-Superhost by Host Scale](images/Quality%20gap%20sh%20vs%20non-sh.png)

These operators represent roughly 25% of the market. They're the newest entrants (mean 2.1 years), expanding rapidly, and apparently tolerating lower ratings because their business model doesn't require quality excellence.

This seems like a host problem, but could easily evolve into a destination problem. Visitors who experience subpar stays may not distinguish between "bad operator" and "bad destination."

---

## The Geographic Dimension

Thessaloniki's STR market isn't uniformly distributed. It clusters dramatically around the tourist core.

### The Concentration Pattern

Distance is measured from the **White Tower / Aristotelous Square midpoint**. It will serve as the focal point of Thessaloniki's tourism activity.

![Distance Category Map](images/Distance%20category%20map.png)

| Zone | Distance | Listings | Share |
|------|----------|----------|-------|
| **Downtown** | <1 km | 1357 | 32.9% |
| **Inner City** | 1-3 km | 2189 | 53.1% |
| **Neighborhoods** | 3-6 km | 487 | 11.8% |
| **Suburban** | >6 km | 91 | 2.2% |

**86% of listings fall within 3km of the White Tower**. The median listing sits at 1.24km from the chosen landmark. That is: most major attractions are within walking reach from almost half the listings. On the other hand, suburban presence is minimal, thereby constraining the statistical validity of analyses conducted on this segment.

### Professional Hosts Concentrate in High-Demand Areas

Large Multi operators systematically target central locations in an effort to capture tourist demand.

![Host Category Distribution by Distance Zone](images/Host%20category%20distribution%20by%20distance%20zone.png)

| Zone | Large Multi Share |
|------|-------------------|
| Downtown | **45%** |
| Inner City | 41% |
| Neighborhoods | 14% |
| Suburban | 13% |

The 3.4× difference between downtown and suburban/neighborhood Large multihost presence reflects economic reality: prime tourist locations justify professional management overhead, while peripheral zones remain the domain of spare-room hosts.

### Downtown's Quality Paradox

Here's where geography gets interesting. Downtown listings, dominated by Large Multi operators who achieve only 32% superhost rate market-wide, somehow deliver a **48% superhost rate**.

![Superhost Rate by Distance Zone](images/Superhost%20rate%20by%20distance%20zone.png)

The answer here is **competition**. The data shows that downtown's intense competitive environment forces all host categories to elevate their quality standards.

![Downtown Quality Uplift by Host Category](images/Downtown%20quality%20uplift.png)

| Host Category | Market Superhost Rate | Downtown Superhost Rate | Uplift |
|--------------|----------------------|------------------------|--------|
| Individual | 39% | 41% | +2% |
| Small Multi | 47% | 46% | -1% |
| Medium Multi | 45% | **58%** | **+13%** |
| Large Multi | 32% | **45%** | **+13%** |

Medium multihosts in downtown achieve a whopping **58% superhost rate**, 13 percentage points above their already high market average. Even Large Multi operators perform significantly better downtown (+13%). The competitive intensity of prime locations forces quality adaptation.

Downtown doesn't just attract professional operators; it improves them.

### Central Adjacency Does Not Imply Good Positioning

Despite geographic centrality, Inner City (1-3km) shows the **lowest location ratings** with the highest variance.

![Location Ratings by Distance Zone](images/Location%20ratings%20by%20zone.png)

The fact that Inner City listings account for more than half of the market, means it combines diverse neighborhoods with varying appeal. Some neighborhoods are great while others are less so, hence the high variance.

Those bad ratings may also signal an expectation mismatch: guests book "central" listings expecting walkability to attractions, then discover practical accessibility challenges.

### Price Positioning by Zone

Geography shapes pricing strategy.

![Price Segment Distribution by Distance Zone](images/Price%20segment%20by%20distance%20zone.png)

| Zone | Premium (€80+) | Budget (<€60) |
|------|----------------|---------------|
| Downtown | **32%** | 46% |
| Inner City | 12% | **72%** |
| Neighborhoods | 20% | 53% |
| Suburban | 43% | 34% |

Downtown commands premium positioning. Inner City is decisively the budget zone. Suburban shows bimodality, likely due to larger properties or unique/ niche offerings that justify distance.

---

## Temporal Dynamics: A Market in Motion

The geographic analysis revealed downtown's current dominance. But Thessaloniki's STR market is young and rapidly evolving. Where is it heading?

### The Youth of the Market

![Listings by First Review Year](images/New%20listings%20by%20year.png)

**Over half of all active listings are less than 2 years old.** The post-pandemic period has seen explosive growth—2022, 2023, and 2024 each set new records for market entries. Early 2025 data suggests this trajectory continues.

| Market Maturity | Time in Market | Share |
|----------------|----------------|-------|
| **New** | <2 years | 52.6% |
| **Growing** | 2-4 years | 22.4% |
| **Mature** | 4-8 years | 20.8% |
| **Established** | >8 years | 4.2% |

This is a young market, still finding its equilibrium.

### Who's Driving the Boom?

The post-pandemic expansion reveals market polarization.

![Market Share by Host Category and Maturity](images/Market%20share%20by%20host%20category%20and%20maturity.png)

| Host Category | Established / Mature (>4yr) | New (<2yr) | Trend |
|--------------|-------------------|------------|-------|
| Large Multi | 25% | **43%** | ↑↑ |
| Individual | 29% | **24%** | ↓ |
| Small + Medium Multi | 46% | **33%** | ↓↓ |

**Large Multi operators nearly doubled their market share** from 2021 onwards as institutional capital enters the sector. Individual hosts also recently rebounded from previous years' share loss, suggesting that Greeks increasingly view tourism as an accessible income stream.

The squeeze is on mid-scale operators. The market appears to be bifurcating: professional platforms versus personal hosting, with shrinking room for hybrid models.

### The Quality Divergence

New listings show dramatically higher rating variance than established ones scoring almost **2.4× greater standard deviation** (σ = 0.51 vs 0.21).

![Quality Variance by Market Maturity](images/Quality%20variance%20by%20maturity.png)

Levene's test confirms this isn't just sampling noise (p < 0.001). Quality predictability declines with market growth: some new entrants excel immediately while others underperform significantly.

### Where Quality Is Eroding

The variance story has a specific shape. An upward trend in premium listing (>€120) quality across host categories emerges, with new listings showing the strongest improvement as hosts compete to attract guests willing to pay premium rates. On the other hand, things seem concerning in the budget segment.

![Price Segment Line Chart](images/Price%20segment%20line%20chart.png)

**Large Multi hosts show clear quality decline in budget categories** (Low and Very Low priced listings) among post-2023 entrants, while Individual and Small/ Medium multihosts maintain stable quality across the same period.

![Large Multi Quality Decline Heatmap](images/Large%20multi%20quality%20heatmap.png)

Zooming into core zones (Downtown and Inner City) for 2021-2025:

![Large Multi Budget Quality - Core Zones](images/Large%20multi%20quality%20decline%20zones.png)

Downtown's Low-budget (€40-60) segment and Inner City's Very Low (<€40) segment both show declining trajectories. Statistical significance is modest (small effect sizes), but the directional finding is consistent: **the "quality uplifter" effect identified in downtown appears to be eroding under volume pressure**.

---

## The Bigger Picture

Thessaloniki's short-term rental market is, by most measures, healthy. High regulatory compliance, competitive dynamics that counter scale advantages, Downtown competition that elevates quality, a diverse host ecosystem serving multiple market segments.

But the data reveals tensions worth monitoring:

**Quality divergence is real.** New listings from Large Multi operators in budget segments show emerging quality concerns: variance is increasing, and mean ratings are declining in core zones. This isn't crisis-level, but it's the kind of trend that compounds over time.

**Scale is rewarded differently than quality.** Large operators achieve market share through volume, but their quality premium is diluted. This creates structural incentives that may not align with destination reputation goals.

**The mid-scale squeeze matters.** Hosts with 2-10 listings achieve the best quality metrics, but they're losing market share to both ends of the spectrum. If this segment shrinks further, the market loses its quality backbone.

**Geography concentrates everything.** 86% of listings cluster within 3km of center. Any quality or saturation issues in downtown ripple through the entire market.

For policymakers, the implications are clear: **host diversity matters**. The optimal tourism ecosystem isn't one dominated by commercial operators or individual hosts, but it's one where mid-scale professionalization thrives alongside them. Competition enforces quality standards, and regulatory oversight prevents concentration from undermining market health.

Thessaloniki's STR market has the pieces in place. The question is whether growth continues to strengthen the ecosystem or it begins to erode it.

---

## Methodology Notes

- **Data source**: Inside Airbnb (June 2025 snapshot).
- **Sample**: 4,124 licensed listings after compliance and data validation filtering.
- **Statistical approach**: Non-parametric tests (Kruskal-Wallis, Mann-Whitney U) with effect size reporting (ε², Cramér's V, rank-biserial correlation).
- **Geographic reference**: White Tower / Aristotelous Square midpoint (40.62962°N, 22.94473°E).
- **Market maturity proxy**: First review date.
- **Limitations**: Revenue estimates based on occupancy proxies; suburban sample (n=91) limits statistical power for that segment; first review date may not precisely capture market entry timing.

---

*This analysis was conducted as part of a data analytics portfolio project. The accompanying Power BI dashboard provides interactive exploration of these findings.*
