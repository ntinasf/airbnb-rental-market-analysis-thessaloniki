Notebook 1 - Regulatory compliance

## Regulatory Compliance Analysis

This notebook analyzes Thessaloniki's short-term rental licensing compliance 
patterns to identify regulatory framework effectiveness and concentration risks.

### Regulatory Background
- **Regular Licenses:** Standard operating permits (required for most properties)
- **Exempt Status:** Policy exemptions for specific property categories
- **Duplicate Licenses:** Single license shared across multiple properties
- **NA/Missing:** Properties operating without clear licensing status

### Research Questions
1. How many listings are unlicensed or falsely claiming exemptions?
2. Is the licensing system achieving its policy objectives of equitable market access while preventing operator concentration?

### Key Finding Preview
> **High compliance, concentrated exemptions.** 97.3% of active listings hold valid licenses, but 2 hosts control 42% of all exemptions and 6 licenses are shared suspiciously across multiple hosts/locations. After filtering 115 non-compliant listings (2.7%), 4,124 regular licensed listings remain for analysis.

## Unlicensed Properties

<Distribution of Estimated Occupancy for Unlicensed Listings barchart>

#### **Main Findings**

* Apart from 4 listings that appear currently active, the rest seem like dormant listings.
* One particular listing is highly active in the market, performing exceptionally well.

## Exempt Licenses

<Property Types Among Exempt Listings barchart>
<Concentration of Exempt Listings by Host piechart>

#### **Main Findings**

* A few hosts hold multiple exemptions, with two hosts controlling almost 21% of all exemptions each (41.6% together)
* No particular geographical pattern that warrants exemptions
* Apart from one listing that is of type "Camper/RV", all the rest are of regular property type
* Minimum availability for most listings is quite high, indicating properties that are not main residences
* Many hosts are not located in Greece
* Many listings have recent reviews, indicating active rentals
* 4 of these hosts are superhosts

## Duplicate Licenses

**Note:** The `location` column is derived by rounding latitude and longitude coordinates to 3 decimal places (approximately 111-meter precision) and concatenating them. While this provides a reasonable proxy for identifying distinct listing locations, results should be interpreted with caution due to potential rounding artifacts and data quality variations.

<table with suspicious duplicate licenses>

#### **Main Findings**

* 6 licenses are shared across more than one host, 5 of them at more than one location
* 74 such listings (28.5% of duplicate licenses) will be dropped from further analysis
* A particular host holds 29 different listings under one license, and 43 listings (16.5% of total duplicate listings) under 6 licenses overall

### **Summary and Recommendations**

* Apart from 4 unlicensed listings that seem currently active, all the rest seem like deactivated listings. Auditing efforts should focus on these active unlicensed listings to ensure compliance.
* Exempt licenses are predominantly held by a few hosts, with two hosts accounting for nearly 40% of all exemptions. This concentration suggests a need for stricter scrutiny of exemption claims to prevent potential misuse.
* Auditing efforts should also focus on listings that share their license across multiple hosts and locations to ensure that license sharing rules are not being abused.

**Note:** The above recommendations are based on the analyzed dataset and should be validated with additional data sources and regulatory guidelines to ensure comprehensive compliance assessment.
Also note that many cases of problematic licensing had been dropped during the data validation phase.


Taking into account the previous findings, we can now separate the listings into regular and non-regular license holders, and base our further analysis on the regular ones.


#####################################
Notebook 2 - Host Type Impact

## Host Ecosystem Analysis: Does Scale Compromise Quality?

Thessaloniki's short-term rental market has grown rapidly, raising questions about whether professional operators enhance or diminish the guest experience. Understanding host ecosystem dynamics is crucial for evidence-based tourism policy.

### Research Questions
1. Do multi-property hosts achieve different guest engagement patterns than smaller operators?
2. Does the current host ecosystem structure benefit the rental market?
3. Is there a "sweet spot" in host portfolio size for optimal performance?

### Host Categorization Framework

| Category | Listings | Profile |
|----------|----------|---------|
| **Individual** | 1 | Casual/occasional hosts, often sharing personal space |
| **Small Multi** | 2-3 | Semi-professional, transitioning to STR business |
| **Medium Multi** | 4-10 | Professional operators, dedicated STR management |
| **Large Multi** | 11+ | Commercial/corporate operators, scaled operations |

### Key Finding Preview
> *Mid-scale professional hosts (2-10 listings) achieve the optimal balance of operational efficiency and guest experience quality. Large commercial operators (11+ listings) show signs of "scale without soul" — higher market share but lower quality scores, suggesting a volume-over-quality approach.*

## Who Controls the Market?

<Distribution of Host Total Listings Count histogram>

<Distribution of Host Total Listings Count boxplot>

<Distribution of Listings by Host Category barchart>

#### **Main Findings**

* Large Multi hosts (11+) dominate with 1,576 listings (38% market share) — the single largest category.
* Individual hosts contribute 1,003 listings (24% market share) — nearly a quarter of supply.
* Medium Multi (4-10) and Small Multi (2-3) are roughly equal: 780 and 765 listings respectively (~19% each).
* Long tail of mega-operators: 1,658 listings come from hosts with 10+ properties.
* The boxplot reveals extreme outliers — some hosts control 200-1,000+ listings (not only in Thessaloniki but globally).
* A single host holds almost 7% of total listings of Thessaloniki.

#### **Comment:** 

There is a clear structure in host types, while the majority of hosts are small operators, the biggest portion of listings is controlled by large multi-property hosts, indicating a trend towards professionalization in the market.

## Does Scale Drive Performance?

<Estimated Revenue Across Host Categories boxplot>

#### **Main Findings**

* Scale does not guarantee higher revenue per listing. 
* Median annual revenue is comparable across all categories, ranging from €1,980 (Medium Multi) to €2,688 (Large Multi). Although statistically significant ($p < 0.001$), the negligible effect size ($\epsilon^2 = 0.007$) indicates that portfolio size has no practical impact on individual listing performance.

<Estimated Occupancy Across Host Categories boxplot>

#### **Main Findings**

* Occupancy shows no meaningful variation across host categories. 
* Despite statistical significance ($p < 0.001$), the negligible effect size ($\epsilon^2 = 0.006$) suggests a competitive market structure where scale provides no inherent advantage in attracting bookings.

### **Comment:** 

One would expect that professional operators (Large Multi) would have higher occupancy due to professional advantages like pricing optimizations tools, more aggressive marketing or better management systems, but this doesn't seem to be the case here.

## Does scale equal quality?

<Listing Age By Host Categories boxplot>

#### **Main Findings**

* Large Multi hosts show younger listings (median 1.6 years) compared to other categories (median 1.9-2.4 years), suggesting recent market entry by commercial operators.
* While the effect size is small ($\epsilon^2 = 0.02$), this pattern indicates growing professionalization of Thessaloniki's STR market in recent years.

### **Comment:** 

The finding supports a narrative that corporate/commercial operators have expanded more recently, which is policy-relevant for:

* Monitoring market concentration trends
* Understanding if regulations prompted or deterred professional entry
* Forecasting future market structure

<Acceptance Rate by Host Type barchart with error bars>

#### **Main Findings**

* Large Multi hosts accept ~5% more requests than the other groups, suggesting that large operators are more professional in responsiveness
* All groups show high acceptance rates (>90%), indicating overall strong acceptance rates
* Maybe a sign of an "accept all" strategy by some large operators? 

<Review score barchart and boxplot by Host Type>

#### **Main Findings**

* Review ratings show a gradual decline as host scale increases, from a median of 4.9 stars (Individual) to 4.7 stars (Large Multi)
* Host category has a moderate to large effect on review scores ($\epsilon^2 = 0.12$)
* The variability in ratings also increases with host scale, indicating less consistent guest experiences among larger operators
* Large Multi hosts ratings drag the average down significantly

#### **Comment:** 

While professional operators show marginally better operational metrics (acceptance rates), guest satisfaction tells the opposite story: Individual hosts achieve the highest ratings (4.84★) with the most consistency, while large multi-listing hosts show both lower scores (4.57★) and greater variability. This 0.27-star gap represents a meaningful quality trade-off that policymakers should consider when evaluating market concentration.

## The Superhost Economy

**Note:** The Superhost status is awarded per host based on consistent high performance across key metrics.
Airbnb evaluates a host's performance every 3 months over the past 12 months for all listings across their account.
The key metrics that must be met are:
* \>90% response rate
* \>4.8 review scores
* <1% cancellation rates
* At least 10 reservations or 3 reservations totaling 100+ nights

<Superhost Achievement by Host Scale barchart>

#### **Main Findings**

* Mid-scale hosts (2-10 listings) achieve highest superhost rates, an indication that "Growth mindset" operators invest in quality to build reputation
* Mid-scale segment may represent optimal balance of professionalization and guest experience quality
* Large Multi hosts have the lowest superhost rate (31,8%), maybe a sign that superhost criteria are harder to meet at scale ?

## Does Quality Pay?

<Superhost Premium by Host Category table>

<The Revenue Gap: Superhost vs Non-Superhost by Host Scale dumbell plot>

<The Quality Gap: Superhost vs Non-Superhost by Host Scale dumbell plot>

<Superhost Premium: Effect Sizes by Variable and Host Category heatmap>

#### **Main Findings**

* Superhost status is a massive revenue differentiator for smaller operators (especially Individual and Small Multi hosts - 3.23x and 3.19x respectively)
* Large Multi non-superhosts is the group with the youngest listings (2.1 years old average), indicating a dynamic where large operators prioritize portfolio expansion
* Large Multi non-superhosts average ratings are significantly lower (4.42★) than other groups, suggesting a quantity-over-quality approach

#### **Comment:** 

Large multi-listing operators (11+ properties) represent Thessaloniki's newest market entrants (median 1.6 years vs. 2.4 years for individuals), suggesting a recent wave of commercial investment in the STR sector. However, this rapid scaling has come at a measurable cost to guest experience.  

Specifically, non-superhost large operators - representing ~25% of the market — deliver the worst guest experience (4.42★), a full 0.3 stars below the overall average. Yet those operators appear to tolerate lower ratings, likely because their business model prioritizes volume over premium services. Or even because quality investment is less rewarded at scale — perhaps guests already expect professionalism from commercial operators. This is backed up by data: superhost status yields only a 1.7x revenue multiplier for large operators, compared to over 3x for smaller hosts. 

Things are different for individual and small multi-property hosts. Quality is not merely a differentiator — it's a survival strategy. The superhost badge serves as a critical lever for revenue growth, enabling smaller operators to compete against commercial scale through reputation and guest loyalty.

### **Summary and Recommendations**

Our analysis suggests that Thessaloniki's STR market performs optimally in the small-to-medium host segment (2-10 listings). These operators achieve:

* The highest superhost rates (45-47%)
* Strong revenue performance
* Quality scores comparable to individual hosts

On the other hand, large commercial operators (11+ listings), while contributing to market supply, show evidence of a "volume over quality" approach that may undermine destination reputation. Their youngest listings, lowest superhost rates, and significantly lower non-superhost review scores (4.42★) suggest rapid, quality-agnostic expansion.

For policymakers, the message is clear: Host diversity matters. Policies that encourage mid-scale professionalization while maintaining quality standards will better serve Thessaloniki's tourism ecosystem than unchecked commercial consolidation.


#####################################
Notebook 3 - geographic performance

## Geographic Performance Analysis: Does Location Drive Success?

Thessaloniki's short-term rental market spans from the city waterfront to emerging residential neighborhoods and suburbs. Understanding how location influences listing performance is essential for both hosts seeking optimal positioning and policymakers managing tourism's spatial footprint.

### Research Questions
1. Does location drive success in Thessaloniki's STR market?
2. Are there distinct performance patterns across geographic zones?
3. Does distance from the tourist core affect guest satisfaction and engagement?

### Geographic Framework

Distance is calculated from the **White Tower / Aristotelous Square midpoint** (40.62962°N, 22.94473°E) — the symbolic and functional heart of Thessaloniki's tourism activity.

| Zone | Distance | Character |
|------|----------|-----------|
| **Downtown** | <1 km | Tourist core: White tower, Aristotelous, Ladadika, Tsimiski str |
| **Inner City** | 1-3 km | Urban residential: Ano Poli, university area, Agiou Dimitriou str |
| **Neighborhoods** | 3-6 km | Suburban-urban mix: Toumba, Kalamaria, Evosmos,  residential zones |
| **Suburban** | >6 km | Peripheral: Chortiatis, airport corridor, eastern suburbs |

### Key Finding Preview
> **Downtown's Triple Convergence**: The city center (<1km) combines professional host concentration (45% Large Multi), highest superhost rates (48%), and premium pricing (31% at €80+). Notably, downtown's competitive environment elevates quality across *all* host categories—Medium Multi hosts achieve 58% superhost rate downtown vs. 45% market-wide. Location ratings show the strongest geographic effect ($\epsilon^2 = 0.19$), confirming guests value walkability to attractions.

## Where Are the Listings?

<map colored by distance category>

<Distribution of Distance from Center histogram and distance category boxplot>

#### **Main Findings**

* Thessaloniki's STR market is heavily concentrated near the city center.
* 86% of listings fall within 3km of the White Tower (Downtown + Inner City).
* Median distance: 1.24 km — half of all listings are within walking distance of the tourist core.
* Suburban presence is minimal: Only 2.2% of listings (91 properties) operate beyond 6km.

#### **Comment:** 

With such strong geographic concentration, any performance differences between zones carry significant market-wide relevance. The sparse suburban sample (n=91) limits statistical power for that category — findings should be interpreted with caution.

## Who Operates Where?

<Host Category Distribution Across Distance Zones horizontal stacked barchart>

#### **Main Findings**

* Large Multi-property operators capture 45% of downtown listings but only 13% of suburban properties — a 3.4x difference. 
* More than half of prime tourist locations are run by professional hosts with multiple listings, indicating a strategic focus on high-demand areas.
* Smaller hosts (Single and Small Multi) dominate away-from-center and suburban zones, likely reflecting second home sharing or investment barriers outside the core.

## Downtown's Quality Paradox

<Superhost Concentration by Distance Zone barchart>

#### **Main Findings**

* Downtown's 48% superhost rate is 10 percentage points above market average.
* Effect size is weak but meaningful (Cramér's V = 0.13) given the binary nature of the superhost variable.
* The other zones more or less follow the same pattern of the overall superhost distribution.

#### **Comment:** 

Large multi hosts hold the majority of listings Downtown (45%), but only 35% of them are superhosts, it is somehow surprising to see that 48% of Downtown listings belong to superhosts.
This small paradox begs for further investigation.

<Downtown Elevates Quality Across All Host Types paired barplot>

#### **Main Findings**

* Medium Multi hosts score 58% superhost rate downtown — a 14% uplift from their overall percentage.
* Large Multi hosts also show significant uplift, from 32% overall to 45% downtown (+13%).
* Individuals and Small Multi hosts don't diverge much from their market-wide percentage.

#### **Comment:** 

There is a clear pattern here: every category performs better downtown than their market average. This means that Downtown's competitive intensity elevates everyone's game.
The highlight is the 58% superhost rate from medium multi operators, which means that they bring their A-game to the prime location.
Equally significant is the 13% uplift for large multi hosts, which suggests that they can achieve quality when competition demands it.
On the other hand, smaller hosts seem to operate close to their ceiling, an indication that downtown smaller hosts don't differ much from those elsewhere.

## Location Economics

<Estimated Revenue by Distance Zone horizontal barchart>

#### **Main Findings**

* Downtown properties yield higher revenue than the rest distance categories, but location alone explains only 1% of this variation, hinting the existence of more confounding variables.

## Guest Location Satisfaction

<Location Ratings by Distance Zone boxplot>

#### **Main Findings**

* Downtown has the highest mean and the lowest variance, a clear sign that guests appreciate being next to the city center. This consistency likely contributes to significantly higher superhost rates.
* Despite being central, Inner City listings get the lowest location ratings combined with big variance.

#### **Comment:** 

Inner city listings show both low ratings and great variance. This is partly explained by the fact that it is the larger group of the three and combines diverse neighborhoods with varying levels of appeal. But it can also signal a condition where central places of interest are not so easily accessible despite being close, thus undermining the overall visitors' experience. Further qualitative research could help unpack guest perceptions in these zone.

## Price Positioning by Zone

<Price Segment Distribution by Distance Zone horizontal stacked barchart>

#### **Main Findings**

* **Downtown commands premium positioning**: 31% of downtown listings price at €80+ (High + Very High), compared to just 12% in Inner City.
* **Inner City is the budget zone**: 72% of listings are under €60, probably serving price-sensitive travelers or extended stays.
* **Suburban bimodality**: 43% of suburban listings target premium segments—likely larger properties or unique offerings that justify distance from center.
* Cramér's V = 0.164 indicates a weak but meaningful association.

### Neighbourhood Overview

The dataset includes 7 administrative municipalities. **Thessaloniki** (the central municipality) dominates with the vast majority of listings, while peripheral municipalities have limited representation. Due to small sample sizes (n < 150 each for peripheral areas), the following summary is **descriptive only** as formal statistical comparisons would lack robustness.

<Neighbourhood Performance Heatmap filled with main performance statistics>

#### **Neighbourhood Performance Summary**

* **Pavlou Mela** seems to be a hidden performer: top superhost rate (52%) very high median revenue and by far the highest occupancy, and all of the above for the second lowest price.
* **Kordelio-Evosmos** shows the highest revenue, value ratings and listing age. A sign that well-established listings in residential areas can perform strongly.
* **Kalamaria** stands out for its high priced listings and strong location ratings, although far from the city center, signifying its appeal as a seaside residential area.
* **Neapolis-Sykeon** has the youngest listings (1.4 years) yet matches top superhost rates (42%), probably a promising growth area.

> *Note: These observations should be validated with larger samples before policy recommendations.

### **Summary and Recommendations**

Diving into Thessaloniki's STR location dynamics, some interesting patterns are revealed: 

* Guest preferences strongly favor downtown proximity as evidenced by location ratings ($\epsilon^2 = 0.19$).
* Downtown positioning explains quality outcomes better than revenue outcomes. High demand for central locations elevates competition, driving hosts to noticeably improve their quality standards and guest experience.
* Large multi-property operators systematically target downtown and inner city listings in an effort to capitalize on tourist demand (~3× Large Multihosts concentration downtown vs. neighborhoods).
* Despite being near the city center, inner city listings show the lowest location ratings, signaling a possible expectation mismatch. These listings mostly serve as the budget zone of the market, with 72% of properties priced under €60, catering to price-sensitive guests.
* Peripheral areas show a more balanced host mix, with smaller operators dominating. 

All the above paint a picture of a market where location significantly influences both host strategy and guest experience. Downtown listings seem to form a self-reinforcing competitive ecosystem where quality is elevated across the board. But policy makers should be cautious about over commercialization of in this part of the city and ensure the viability of residential life and market balance.
Efforts should also focus on raising the standards of listings in inner city, both in terms of quality/guest experience and city center accessibility (e.g. transport links, walkability).


#####################################
Notebook 4 - temporal dynamics

## Market Evolution & Temporal Dynamics: Where Is Thessaloniki's STR Market Heading?

The geographic analysis revealed downtown's current dominance—professional hosts, premium pricing, and quality competition. But Thessaloniki's short-term rental market is young and rapidly expanding. This notebook traces **temporal patterns** to understand how the market is evolving and what trajectory it's on.

### Research Questions
1. How has the STR market expanded over time? Which years saw the biggest influx of new listings?
2. Are new listings concentrating in existing hotspots or expanding to peripheral areas?
3. Which host categories are driving recent growth? Are large operators accelerating their expansion?
4. What price segments are new entrants targeting—premium or budget?
5. Do newer listings maintain the quality standards of established ones, or is rapid growth diluting market quality?

### Analytical Framework: Market Maturity

To analyze temporal patterns, listings are categorized by their **time in market** based on first review date:

| Category | Time Since First Review | Market Profile |
|----------|------------------------|----------------|
| **New** | < 2 years | Recent entrants, still establishing reputation |
| **Growing** | 2-4 years | Maturing listings, building review history |
| **Mature** | 4-8 years | Established presence, stable performance |
| **Established** | > 8 years | Market veterans, pre-dating STR boom |

> **Note**: First review date serves as a proxy for market entry. Listings without reviews are excluded from temporal analysis as their entry timing cannot be determined.

### Key Finding Preview
> *Thessaloniki's STR market is undergoing a **quality divergence** driven by rapid post-pandemic expansion. While the premium segment self-regulates effectively, the budget segment—particularly among new Large multihost operators—shows signs of emerging quality concerns that warrant attention.*

## A Young Market

<Listings by First Review Year histogram and Market Maturity barplot>

#### **Main Findings**

* Thessaloniki's STR market is a young, rapidly evolving market: Over half of all active listings are less than 2 years old.
* There is a post-pandemic explosion of new listings. Particularly from 2022 onwards, every new year has been a record-breaker for new market entrants.
* There was a slow but steady growth in the market just before the pandemic, with listings gradually increasing from 2015 to 2019.
* Listings from 2025 are set to match or even exceed those from 2024, indicating continued strong growth (the data stops at June 2025).

## Where Is Growth Happening?

<Absolute Growth by Distance Zone line chart>

<Relative Market Share by Distance Zone line chart>

#### **Main Findings**

* Inner City and Downtown listings continue to lead the market growth pattern.
* Expansion for peripheral areas remains marginal despite the post-pandemic boom.
* Since 2020, each distance zone's share has converged toward its overall market share — geographic distribution has stabilized.
* Early volatility in market share is not meaningful due to small sample sizes.

## Who's Driving the Boom?

<Host Category Composition by Market Maturity horizontal stacked barplot>

#### **Main Findings**

* Large Multi hosts capture 43% of new market entries, nearly doubling their share among pre-2021 listings (25%). 
* Individual hosts rebound in the market (24% from 20%) after losing a significant market share in previous years, suggesting a renewed interest in casual hosting
* Small and medium multihosts see their share squeezed out as the market evolves (46% Established -> 39% Mature -> 33% New).

#### **Comment:** 

The post-pandemic boom reveals market polarization driven by tourism optimism. Two forces are simultaneously reshaping Thessaloniki's STR landscape:

* Large multihosts almost doubled their market share, a sign that institutional capital enters the sector.
* Individual hosts rebounded from their COVID-era decline, as Greeks increasingly view tourism as an accessible income stream.

As mid-scale operators are losing ground,the market seems to bifurcate into professional platforms vs. personal hosting, with little room for hybrid models.

## Price Segment Targeting

<Price Category Composition by Market Maturity horizontal stacked barplot>

#### **Main Findings**

* Price segment distribution has remained largely stable across market cohorts.
* The Low (€40-60) segment shows modest gains at the expense of Very Low (<€40), suggesting slight upward price positioning among newer entrants.

## The Quality Divergence

<Quality Variance by Market Maturity boxplot>

#### **Main Findings**

* **Variance increases significantly with market age**: New entrants show 2.4x bigger standard deviation than mature listings (σ = 0.21 vs 0.51), a sign that quality predictability declines.
* Despite the increase in variability, median rating is slightly higher in newer listings, a pattern that points to diverging quality trajectories between hosts - some excel immediately while others underperform.

### Quality trajectory between host groups and price segments

<Quality by Host Type Across Age Cohorts grouped barplot>

#### **Main Findings**

* While all host categories maintain relatively stable mean ratings over time, more recent listings from large multihosts show a clear downward trend.

Let's zoom in to large multihost ratings over time by price segment.

<Large Multi Quality Decline heatmap filled with mean ratings by market maturity and price segment>

#### **Main Findings**

* The decline in ratings among new large multihost listings is driven by budget (low and very low) priced listings.


<Quality by Price Segment and Host Type (Post-2021 Entrants) small multiples line charts>

#### **Main Findings**

* Large multihosts show steep quality decline during the past two years for both budget categories (Low and Very Low).
* At the same time, Individual and small multihosts maintain stable quality levels across all budget categories.
* All host types converge to a 4.9 - 5 for Very High priced listings, a sign that a serious competition for premium guests is emerging across the board. (medium effect size)

#### **Comment:** 

Zooming in to post-pandemic years the quality decline trend for large multihosts gets clearer. During the first years there was actually a slight increase in ratings but from 2023 onwards the decline becomes evident for both budget categories. it would be interesting to see which areas were most affected by this trend.

<Large Multi Budget Quality: Core Zones (2021-2025) small multiples for Downtown and Inner City>

#### **Main Findings**

* Downtown Low (€40-60) budget listings show an apparent decline in recent years, both for mean and median values.
* Inner City shows the same trajectory but only for the Very Low budget category.

#### **Comment:** 

The city's premium core (Downtown) appears to be losing its quality shield among budget-friendly listings. This suggests that the advantage of location alone cannot compensate for operational corner-cutting. The "quality uplifter" effect we identified earlier appears to be eroding under volume expansion. Although the results are statistically modest (small effect size), this directional finding points to a worrying trend for the market's long-term quality standards.

### **Summary and Recommendations**

The most notable findings from this temporal analysis are:

* Thessaloniki's STR market is a young and growing one: Over 50% of listings are less than 2 years old with signs of further expansion.
* Large Multi operators almost doubled their market share (25% → 43%) in the post-pandemic period, while mid-scale hosts lost ground.
* New listings show 2.4x the rating variability of established ones: Some new entrants excel while others underperform.
* Premium hosting thrives among every host group while budget segment shows signs of quality erosion, particularly among new Large multihost operators.

For policymakers, these findings suggest a nuanced approach:

The divergence in quality trajectories warrants monitoring. New budget-friendly listings from commercial operators call for early attention, especially in core tourist zones, as lower quality services could undermine the destination's growing reputation.


#####################################