
LEGITIMATE LISTINGS OVERVIEW - KEY INFORMATION CATEGORIES
Market Health Indicators (KPI Cards)

Total Legitimate Listings: Count and percentage of total market
Active Properties: Properties with recent guest activity (reviews in last 12 months)
Average Occupancy Rate: Based on estimated_occupancy_l365d
Revenue Concentration: Geographic distribution of tourism economic impact

Geographic Intelligence

Map: Legitimate listings distribution across Thessaloniki
Neighborhood Performance: Revenue per neighborhood, listing density
Property Type by Area: Show how different areas serve different tourism segments
Price Ranges: Geographic variation in pricing strategies

Host Ecosystem Composition

Individual vs Multi-Property Hosts: Foundation for H3 analysis
Superhost Distribution: Quality indicator across the market
Host Tenure: Experience levels in the legitimate market
Response Rate Patterns: Service quality baseline

Guest Satisfaction Baseline

Review Score Distribution: Overall satisfaction patterns
Guest Capacity: Market ability to accommodate different group sizes
Availability Strategies: How hosts manage their properties
Review Activity: Guest engagement and repeat business indicators

Seasonal Activity Patterns

Year-Round vs Seasonal Properties: Market stability indicators
Peak Season Performance: Tourism demand patterns
Price Flexibility: Dynamic vs static pricing adoption

STRATEGIC FOCUS FOR TOURISM AUTHORITY
This overview page should answer: "What does a healthy, compliant short-term rental market look like in Thessaloniki?"
Key metrics that matter for policy:

Market concentration (is tourism revenue spread across neighborhoods?)
Host diversity (healthy mix of individual and professional hosts?)
Guest satisfaction (are visitors having positive experiences?)
Property utilization (efficient use of tourism infrastructure?)

Visual suggestions:

Use your established 4-KPI card format for consistency
Include a comprehensive map showing legitimate property distribution
Property type breakdown (similar to previous pages)
Host composition analysis (individual vs multi-property)

This legitimate market overview becomes the baseline against which you'll test your hypotheses about optimal host types (H3), performance drivers (H4), and geographic strategies (H5). It demonstrates market health before diving into optimization opportunities.
<hr>
order = CATEGORY_ORDER
categories = [] # replace with constants defined in the notebook,
replace repeated code (visuals, test, with functions)

<hr>
for report


estimated revenue: Document the price distribution in your technical notebook with a note that "pricing strategies are consistent across host types, confirming revenue differences reflect operational performance rather than pricing positioning.

acceptance rate
"Professional hosts (4+ listings) demonstrate marginally higher operational standards, with ~5 percentage points higher acceptance rates, suggesting more systematic management practices."

superhost opening
"The Sweet Spot of Host Professionalization"

Small and medium multi-listing hosts (2-10 listings) achieve the highest superhost rates (45-47%), significantly outperforming both individual hosts (39%) and large commercial operators (32%). This suggests a "growth mindset" phase where operators are actively investing in guest experience to build their business reputation.

Individual hosts, while providing authentic experiences, may lack the motivation to pursue platform-specific metrics. Large operators, managing 11+ listings, appear to prioritize operational efficiency over personalized service, resulting in fewer superhost certifications.

This pattern has implications for market quality: the mid-scale segment (2-10 listings) appears to optimize for both growth AND guest satisfaction, making them potentially the most desirable host category from a tourism quality perspective.

But does superhost status translate to tangible performance advantages? The following analysis explores whether this quality investment pays off...
Transition: "But does this quality certification translate to revenue? Let's examine the superhost premium..."

###############
geographic performance

est revenue

"Downtown properties earn ~30% more than other zones at median (€3,192 vs €2,268), but location explains only 1.5% of revenue variation (η² = 0.015). This suggests operational factors — host professionalization, pricing strategy, quality — matter far more than geography alone."

host cat

"Thessaloniki's professional hosting sector is spatially concentrated: Large Multi-property operators capture 45% of downtown listings but only 13% of suburban properties — a 3.4x difference. This center-periphery gradient reflects economic realities: prime tourist locations justify the overhead of professional management, while peripheral neighborhoods remain the domain of individual hosts sharing spare rooms or second homes. For policymakers, this means regulatory focus on central districts would address the majority of commercial STR activity."

superhost status

This finding complements your host_category × distance finding perfectly:

Downtown paradox: 45% Large Multi hosts (lowest superhost rates per H3) yet 48% overall superhosts
Implication: Downtown's smaller operators must have exceptionally high superhost rates to pull up the zone average
Policy angle: Downtown's quality floor is higher — even non-professional hosts meet superhost standards there
This creates a compelling narrative: "Downtown attracts both professional operators AND quality-conscious individuals — it's where the best compete."

"Downtown's 48% superhost rate initially puzzled us — how does a zone dominated by Large Multi operators (45% of listings, 32% superhost rate market-wide) achieve such quality density? The answer: downtown's competitive pressure forces adaptation. Large Multi operators in downtown achieve 45% superhost status — 13 percentage points above their market average. But the real surprise is Medium Multi hosts, who reach 58% superhost rate downtown — suggesting mid-sized professional operators combine the quality orientation of smaller hosts with the operational efficiency of larger ones. Downtown doesn't just attract professional operators; it improves them."

Include this finding — it's one of the most actionable insights:

For hosts: Medium Multi scale (4-10 listings) may be the optimal portfolio size for quality AND presence in prime locations
For policymakers: Downtown's competitive dynamics self-regulate quality; peripheral areas lack this natural pressure
Connects H3 + H5: Shows host type effects are geographically contingent


review scores location

"Downtown's location rating advantage (4.87 vs 4.63 Inner City) with the tightest variance (σ=0.26) creates a quality floor that makes superhost maintenance easier. Hosts in prime locations face fewer guest complaints about 'being far from attractions' — a common rating killer. This structural advantage partly explains why even Large Multi operators achieve 45% superhost rate downtown versus 32% market-wide."


<hr>
for dashboard
add gradient rating effect on 2nd page plus error bars