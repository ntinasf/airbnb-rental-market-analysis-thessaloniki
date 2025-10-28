# Aristotelous Square: 40.6333° N, 22.9417° E
# This is Thessaloniki's central landmark and tourist hub
```

**What to create:**
- `distance_to_center_km` column
- Location categories: Historic Center (<1km), Inner City (1-2km), Neighborhoods (2-5km), Suburban (>5km)

**Checkpoint questions:**
- What's the distance range? (expect 0-10km for most listings)
- Are listings concentrated near center or distributed?
- Any extreme outliers? (>15km might be data errors)

---

#### **1.2 Neighborhood-Level Summary Statistics**

**Create aggregated dataset:**
```
For each neighborhood, calculate:
- Number of listings (market size)
- Average price
- Median price (less affected by outliers)
- Average revenue
- Average rating
- % Superhost properties
- Host category distribution
- Average distance from center
```

**Checkpoint questions:**
- How many neighborhoods have 50+ listings? (focus on these)
- What's the price range across neighborhoods? (expect 2-3x variation)
- Do any neighborhoods stand out as obvious outliers?

**REPORT BACK:** Share top 10 neighborhoods by listing count with their average price and distance from center

---

### **PHASE 2: EXPLORATORY SPATIAL ANALYSIS (3-4 hours)**
**Time-box: Complete by end of Day 2**

#### **2.1 Price Distribution by Geography**

**Key analyses:**
1. **Distance decay pattern**
   - Scatter plot: distance_to_center_km vs price
   - Calculate correlation coefficient
   - Expected: Negative correlation (center = expensive)

2. **Neighborhood ranking**
   - Top 10 most expensive neighborhoods
   - Top 10 cheapest neighborhoods
   - Compare average ratings between groups

3. **Spatial price patterns**
   - Simple scatter: longitude vs latitude, colored by price
   - Identify geographic clusters visually
   - Look for "price islands" (expensive pockets in cheap areas)

**REPORT BACK:** 
- Distance-price correlation coefficient and pattern
- Top 5 most/least expensive neighborhoods with avg price
- Any surprising geographic patterns (e.g., cheap listings in center)

---

#### **2.2 Performance Patterns by Location**

**Key analyses:**
1. **Revenue vs distance**
   - Do central properties earn more despite competition?
   - Or do peripheral properties win on volume?

2. **Occupancy vs distance**
   - Tourist preference for central location?
   - Longer stays in peripheral areas?

3. **Value score calculation**
```
   value_score = review_scores_rating / (price / median_price)
```
   - Where are the "best value" neighborhoods?
   - Do cheap neighborhoods over-deliver on quality?

**REPORT BACK:**
- Do central or peripheral neighborhoods have higher revenue?
- Value leaders: Which neighborhoods offer best quality-to-price ratio?
- Any location where high price ≠ high quality?

---

#### **2.3 Neighborhood Profiling**

**Identify distinct neighborhood types:**

Look for patterns like:
- **Premium Tourist Zone**: High price, high rating, lots of superhosts
- **Budget Tourism**: Low price, decent rating, individual hosts dominate
- **Overpriced Areas**: High price, mediocre ratings
- **Hidden Gems**: Medium price, excellent ratings
- **Commercial Zones**: Large multi-hosts concentrated

**Create 2x2 matrix:**
```
           High Price  |  Low Price
High Quality    A     |     B (Best Value!)
Low Quality     C     |     D (Avoid)
```

**REPORT BACK:**
- How many neighborhoods fall into each quadrant?
- Name 2-3 neighborhoods in each category
- Which category has most listings?

---

### **PHASE 3: STATISTICAL VALIDATION (4-5 hours)**
**Time-box: Complete by end of Day 3**

#### **3.1 Primary Hypothesis Tests**

**Test 1: Price differences across location types**
```
ANOVA: Does price differ significantly by location_type?
H0: Mean price is equal across Historic/Inner/Neighborhoods/Suburban
Method: One-way ANOVA + post-hoc tests
Effect size: Eta-squared
```

**Test 2: Distance-price relationship**
```
Correlation: distance_to_center_km vs price
Methods: Pearson (linear) and Spearman (monotonic)
Test: Is correlation statistically significant?
```

**Test 3: Neighborhood performance differences**
```
ANOVA: Do top 10 neighborhoods differ in revenue?
Focus on neighborhoods with 50+ listings
Check if differences are statistically significant
```

**REPORT BACK:**
- ANOVA p-values and effect sizes
- Which location comparisons are significantly different?
- Correlation coefficients with interpretation

---

#### **3.2 Integration Tests (Connect H1 & H3)**

**H3 Connection: Host types by geography**
```
Question: Do host categories distribute differently by location?
Test: Chi-square test of independence
Cross-tab: neighborhood_type vs Host_Category

Expected: Large Multi hosts concentrated in tourist center?
```

**H1 Connection: Licensing issues by geography**
```
Question: Do licensing issues cluster geographically?
Analysis: 
- Map exempt licenses by neighborhood
- Map duplicate licenses by neighborhood
- Test if concentration differs from baseline

Expected: Exemptions clustered? Duplicates spread out?
```

**Superhost geography (H3 extension)**
```
Question: Does superhost distribution vary by location?
Test: Chi-square or proportion test
Compare: % superhosts in center vs periphery

Expected: Tourist areas have higher superhost %?
```

**REPORT BACK:**
- Do Large Multi hosts dominate certain neighborhoods?
- Are H1 licensing issues geographically concentrated?
- Superhost % by location type with statistical test

---

### **PHASE 4: INSIGHT SYNTHESIS (2-3 hours)**
**Time-box: Complete by Day 4**

#### **4.1 Create Neighborhood Typology**

**Classify neighborhoods into strategic categories:**

Example framework (adjust based on your findings):

**Type A: Premium Tourist Core**
- Characteristics: High price, high occupancy, superhosts, small footprint
- Host profile: Mix of all types, superhost-heavy
- Example neighborhoods: [Based on your data]
- Business implication: Quality competition, saturation risk

**Type B: Value Tourism Belt**
- Characteristics: Medium price, high value score, individual hosts
- Host profile: Dominated by 1-2 property hosts
- Example neighborhoods: [Based on your data]
- Business implication: Growth opportunity, support needed

**Type C: Residential Integration**
- Characteristics: Lower price, authentic experience, scattered distribution
- Host profile: Individual hosts, longer stays
- Example neighborhoods: [Based on your data]
- Business implication: Community balance focus

**Type D: Commercial Concentration**
- Characteristics: Large multi-hosts, variable quality
- Host profile: 4+ property operators dominate
- Example neighborhoods: [Based on your data]
- Business implication: Regulatory attention needed

---

#### **4.2 Generate Policy Recommendations**

**Framework: Geographic-specific strategies**

For each neighborhood type, recommend:
1. **Regulatory approach** (from H1 insights)
2. **Host support strategy** (from H3 insights)
3. **Tourism development priority** (from H5 insights)

**Example structure:**
```
Premium Tourist Core Neighborhoods:
- Regulation: Strict quality standards, monitor over-tourism
- Host support: Superhost incentive programs
- Development: Maintain quality, limit quantity
- H1 connection: Focus enforcement on duplicate licenses here
- H3 connection: Leverage high superhost concentration

Value Tourism Belt Neighborhoods:
- Regulation: Simplified licensing for individual hosts
- Host support: Training programs, marketing assistance
- Development: Infrastructure improvements
- H1 connection: Fast-track legitimate single-property licenses
- H3 connection: Encourage Individual → Small Multi progression
```

---

### **PHASE 5: VISUALIZATION PREPARATION (1-2 hours)**
**Time-box: Day 4-5**

#### **5.1 Identify Key Visuals for Power BI**

**Must-have visualizations:**

1. **Geographic Price Map**
   - Bubble map: lat/long with price as color
   - Size: revenue or number of listings
   - Interactive filter: Host category, license status

2. **Neighborhood Performance Table**
   - Top 15 neighborhoods
   - Columns: Listings, Avg Price, Avg Revenue, Avg Rating, % Superhost
   - Sortable, conditional formatting

3. **Distance from Center Analysis**
   - Scatter plot: distance vs price with trend line
   - Second scatter: distance vs revenue
   - Show correlation coefficient

4. **Neighborhood Typology Matrix**
   - 2x2 scatter: Price (x) vs Rating (y)
   - Each dot = neighborhood
   - Size = number of listings
   - Quadrant labels

5. **Host Distribution by Location**
   - Stacked bar: Location type vs Host Category %
   - Shows H3 connection

6. **H1 Integration: Licensing Heat Map**
   - Map showing exempt/duplicate concentration
   - Overlay with high-revenue neighborhoods

**REPORT BACK:**
- Which of these visuals will tell the strongest story?
- Any additional visualizations from your exploratory analysis?
- What's the "wow" finding that should be featured?

---

## 🎯 DECISION POINTS & PIVOTS

### **After Phase 1 (Foundation):**
✅ **If distance-price correlation strong (r > 0.4):** Proceed as planned  
⚠️ **If weak correlation (r < 0.3):** Pivot to neighborhood-specific analysis, less emphasis on distance  
🚫 **If no geographic patterns:** Focus on other differentiators (property type, host category by location)

### **After Phase 2 (Exploratory):**
✅ **If clear neighborhood clusters:** Deep dive into typology  
⚠️ **If patterns weak:** Focus on top performers vs bottom performers (simpler story)  
🚫 **If homogeneous market:** Document finding, emphasize H1/H3 integration

### **After Phase 3 (Statistical):**
✅ **If p < 0.05 with meaningful effect sizes:** Full analysis report  
⚠️ **If marginal significance:** Present as trends with caveats  
🚫 **If null results:** Document appropriately (like H4), focus on descriptive insights

---

## ⏱️ TIME BUDGET

| Phase | Time Limit | Deliverable | Report Back |
|-------|-----------|-------------|-------------|
| **Phase 1: Foundation** | 2-3 hours | Distance calcs, neighborhood summary | Top 10 neighborhoods table |
| **Phase 2: Exploration** | 3-4 hours | Patterns, value scores, profiling | Key correlations & neighborhood types |
| **Phase 3: Statistics** | 4-5 hours | ANOVA, correlations, integrations | Test results & effect sizes |
| **Phase 4: Insights** | 2-3 hours | Neighborhood typology, policy recs | Strategic framework |
| **Phase 5: Viz Prep** | 1-2 hours | Visual designs, data for Power BI | Visual priorities |
| **TOTAL** | **12-17 hours** | **H5 complete** | **Ready for Power BI** |

**Spread across 4-5 days with natural checkpoints**

---

## 🎯 SUCCESS CRITERIA

### **Minimum Viable H5:**
- ✅ Distance-price relationship quantified (correlation + p-value)
- ✅ Top 10 neighborhoods profiled with key metrics
- ✅ Statistical test (ANOVA) confirming geographic differences
- ✅ Integration with H1 or H3 (at least one clear connection)
- ✅ 3-4 actionable policy recommendations

### **Strong H5 (Portfolio-worthy):**
- ✅ All of above PLUS:
- ✅ Neighborhood typology framework (3-4 distinct types)
- ✅ Integration with BOTH H1 and H3
- ✅ Value score analysis identifying hidden gems
- ✅ Geographic visualization data prepared
- ✅ Clear business narrative connecting all findings

---

## 📊 KEY QUESTIONS TO ANSWER

### **Business Questions (Tourism Authority wants to know):**

1. **Pricing Strategy:**
   - Which neighborhoods command premium prices and why?
   - Where are the overpriced markets (high price, low satisfaction)?
   - Where are the value opportunities (underpriced quality)?

2. **Development Strategy:**
   - Which neighborhoods should receive tourism infrastructure investment?
   - Where should new licenses be prioritized?
   - Which areas are oversaturated vs underdeveloped?

3. **Regulatory Focus:**
   - Where should compliance enforcement concentrate? (H1 connection)
   - Which neighborhoods need host support vs regulation?
   - Where are commercial operators displacing individuals? (H3 connection)

4. **Market Balance:**
   - Is tourism too concentrated geographically?
   - Which neighborhoods maintain best host diversity?
   - Where is the Individual vs Large Multi balance optimal?

### **Analytical Questions (For statistical rigor):**

1. **Primary:** Is geographic location a significant predictor of price/performance?
2. **Secondary:** Do neighborhoods cluster into distinct profiles?
3. **Integration:** How do H1 (licensing) and H3 (host types) patterns vary geographically?
4. **Practical:** What's the expected price premium per km from center?

---

## 🔗 INTEGRATION STRATEGY

### **Connect H5 findings to previous hypotheses:**

**H1 (Regulatory) × H5 (Geographic):**
```
Question: Where are exempt/duplicate licenses concentrated?
Analysis: Map licenses by neighborhood
Insight: "Exemptions cluster in [neighborhood], requiring targeted policy reform"
```

**H3 (Host Ecosystem) × H5 (Geographic):**
```
Question: Where do different host types thrive?
Analysis: Host category % by location type
Insight: "Large Multi hosts concentrate in tourist center (X%), 
          while peripheral neighborhoods favor Individual hosts (Y%)"
```

**Combined Narrative:**
```
"Thessaloniki's short-term rental market shows distinct geographic 
stratification. The [neighborhood] tourist core supports professional 
operators (H3 Large Multi hosts) commanding €X premiums (H5 price analysis), 
but also concentrates Y% of licensing irregularities (H1 exempt patterns). 
Meanwhile, peripheral neighborhoods demonstrate the optimal Individual/Small 
Multi host balance (H3 sweet spot) with stronger value propositions (H5 
quality-to-price ratios)."