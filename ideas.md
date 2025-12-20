
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

Large multi-property operators accept almost everything (98.4%) because:

Volume over selectivity - They're running a numbers game, not curating guest experiences
Lower standards - They don't turn away "risky" bookings that individual hosts might decline
Efficiency-driven - Less time spent evaluating each request, just accept and move on
No personal stake - It's not their home, so less concern about who stays
Meanwhile, individual hosts are more selective (92.3%) because:

They care about guest fit
They may decline last-minute requests or short stays
They protect their property/neighborhood relationships
The Irony
High acceptance rate + Low quality scores = "Accept everyone, deliver less"

This is the opposite of a superhost approach (which emphasizes selective quality).


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


#########################

temporal dynamics

distance cat - nothing important to note
"Geographic distribution of listings has remained remarkably stable across market cohorts, with Inner City consistently absorbing the majority of new supply regardless of entry timing (Cramér's V = 0.05, negligible)."

host category

"Who's Driving the Boom?"

Thessaloniki's STR market has undergone structural transformation. Among listings entering pre-2017 (Established), host types were relatively balanced. But the post-COVID surge tells a different story: Large Multi-property operators now capture 43% of new market entries, nearly double their share among veteran listings.

Interestingly, Individual hosts show a U-shaped pattern—declining during 2020-2022 but rebounding strongly in 2023-2024, suggesting renewed interest from Greeks seeking tourism income. The losers? Mid-scale operators (2-10 listings), squeezed between professional scale and authentic appeal.

price cat

"Price segment distribution has remained stable across market cohorts, with Low (€40-60) consistently representing ~40-45% of listings regardless of entry timing (Cramér's V = 0.05, negligible)."

host is superhost




<hr>
for dashboard
add gradient rating effect on 2nd page plus error bars



####################
now that we have a full story lets build a blueprint for the overall report and an executive one. For the executive report i dont know much, and even maybe the power bi dashboard can even replace it, so i am counting on your experience and opinion if i should bulid it and how. For the analytical report i want to glue the stories told by the notebooks in one coherent story that would be worthy of publishing to my personal portfolio website. Here i dont know the right analogy of statistical jargon and analytic depth vs accessible-to-everyone storytelling. i am a big fan of scott galloway's data story telling and writting (if you are aware of his style show some glimpses of it, dont over do it) so i was thinking if something similar would word in this case, ie tell the story nicely but also include statistical validity statements (like effect sizes, variance explained, sample sizes) followed by the right visual to communicate it. So if you agree on this approach, i want you to create on the main directory a report.md file that draws context from the notebooks regulatory_compliance.ipynb, host_type_impact.ipynb, geographic_performance.ipynb, temporal-dynamics.ipynb and crafts a story built as i suggested above. there should an opening paragraph with general info, and then smooth transittions to the topics covered by the notebooks, in the order provied. Make sure to include  clear and clever headings for each topic change and to incorporate the business questions answered in the flow. Use place holders for visuals and statistically important number for me to fill. Tailor the suggestion into the narrative and do an approprite closing that paintsa  biggger picture enhanced by our findings. There will be more iterations on this file to improve it but do your best so that we start from the best place possible. If you agree with the executive summary create a separate md that fits your likining. finally dont use the h3, h5 etc namings, those were just for our convenience.



########################
 host dynamics comment

 The Professionalization Paradox
Large multi-listing operators (11+ properties) represent Thessaloniki's newest market entrants (median 1.6 years vs. 2.4 years for individuals), suggesting a recent wave of commercial investment in the STR sector. However, this rapid scaling has come at a measurable cost to guest experience.

The data reveals a troubling pattern:

Lowest Quality Floor: Non-superhost large operators deliver the worst guest experience in the market (4.42★), a full 0.37 stars below individual non-superhosts. This isn't statistical noise—it represents a large effect size (r=-0.60).

Diminished Returns on Quality: While superhost status delivers a 3.2x revenue multiplier for individual hosts (€5,280/year premium), large operators see only a 1.7x multiplier (€2,400/year). Quality investment is less rewarded at scale—perhaps because guests already expect professionalism from commercial operators.

Volume Over Excellence: With the lowest superhost rate (32%) despite controlling 38% of listings, large operators appear to prioritize portfolio expansion over guest satisfaction. The market is rewarding this strategy with marginally higher occupancy (78 vs. 75 days), but at what cost?

Is This Bad for the Market?
Yes, for several reasons:

Guest Experience Degradation: The 4.42★ floor from large non-SH operators drags down Thessaloniki's overall destination quality
Unfair Competition: Individual hosts who invest in superhost status earn 3x more, but compete against commercial operators who achieve acceptable returns with lower quality
Sustainability Risk: Young listings with poor reviews have higher churn risk—if these operators exit, it could destabilize neighborhoods with high concentrations
Tourism Reputation: Visitors who experience these lower-quality properties may not distinguish between "bad operator" and "bad destination"

-- recs 
Policy Recommendation Framework
Based on your findings, I'd suggest framing recommendations around:

1. Quality Incentives for Scale
Tie licensing renewal to minimum review thresholds (e.g., 4.5★)
Progressive requirements: 1-3 listings = baseline, 4-10 = enhanced, 11+ = premium standards
2. Superhost Recognition
Tourism authority could publicly highlight superhosts
Create a "quality tier" system for tourist-facing marketing
3. Market Monitoring
Track the quality trajectory of new large operators
Flag hosts with rapid expansion + declining reviews