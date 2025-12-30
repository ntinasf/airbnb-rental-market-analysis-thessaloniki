# Power BI DAX Formulas
## Thessaloniki Airbnb Market Analysis Dashboard

This document contains all DAX measures and calculated columns used in the Power BI dashboard for analyzing Thessaloniki's short-term rental market.

---

## Table of Contents

1. [Base Table: listings_cleaned](#1-base-table-listings_cleaned)
2. [Regular License Listings](#2-regular-license-listings)
3. [Non-Regular License Listings](#3-non-regular-license-listings)

---

## 1. Base Table: `listings_cleaned`

*Contains all listings before license filtering.*

### Core Metrics

```dax
Total Listings (All Data) = 
COUNTROWS(listings_cleaned)
```

```dax
Compliance Rate = 
DIVIDE([Regular Listings], [Total Listings (All Data)], 0)
```

---

## 2. Regular License Listings

*Table: `listings_regular_license` — Contains only listings with valid, unique license numbers.*

### 2.1 Basic Counts

```dax
Regular Listings = 
COUNTROWS(listings_regular_license)
```

### 2.2 Geographic Distribution

**Core Concentration %** — Percentage of listings within 3km of city center

```dax
Core Concentration % = 
VAR CoreCount = 
    CALCULATE(
        COUNT(listings_regular_license[id]),
        listings_regular_license[distance_cat] IN {"Downtown (<1km)", "Inner City (1-3km)"}
    )
VAR TotalCount = 
    CALCULATE(
        COUNT(listings_regular_license[id]),
        REMOVEFILTERS(listings_regular_license[distance_cat])
    )
RETURN
    DIVIDE(CoreCount, TotalCount)
```

### 2.3 Host Category Analysis

**Host Category %** — Distribution by host portfolio size

```dax
Host Category % = 
VAR CurrentCount = COUNT(listings_regular_license[id])
VAR TotalCount = 
    CALCULATE(
        COUNT(listings_regular_license[id]),
        ALL(listings_regular_license[host_category], listings_regular_license[host_sort]),
        listings_regular_license[host_category] <> BLANK()
    )
RETURN
    DIVIDE(CurrentCount, TotalCount)
```

**Large Multi Downtown Percentage** — Share of large operators (11+ listings) in downtown

```dax
Large Multi Downtown Percentage = 
CALCULATE(
    DIVIDE(
        CALCULATE(
            COUNTROWS(listings_regular_license),
            listings_regular_license[host_category] = "Large Multi (11+)"
        ),
        COUNTROWS(listings_regular_license)
    ),
    listings_regular_license[distance_cat] = "Downtown (<1km)"
)
```

**Large Multi Share Rise** — Growth ratio of large operators (new vs mature markets)

```dax
Large Multi Share Rise = 
VAR NewShare = 
    CALCULATE(
        DIVIDE(
            CALCULATE(
                COUNT(listings_regular_license[id]), 
                listings_regular_license[host_category] = "Large Multi (11+)"
            ),
            COUNT(listings_regular_license[id])
        ),
        listings_regular_license[market_maturity] = "New (<2yr)"
    )
VAR MatureShare = 
    CALCULATE(
        DIVIDE(
            CALCULATE(
                COUNT(listings_regular_license[id]), 
                listings_regular_license[host_category] = "Large Multi (11+)"
            ),
            COUNT(listings_regular_license[id])
        ),
        listings_regular_license[market_maturity] = "Mature (4-8yr)"
    )
RETURN
    DIVIDE(NewShare, MatureShare)
```

```dax
Large Multi Share Rise Text = 
FORMAT([Large Multi Share Rise], "0.0") & "x"
```

### 2.4 Superhost Metrics

**Superhost Count & Percentage**

```dax
Superhost Count = 
CALCULATE(
    COUNTROWS(listings_regular_license),
    listings_regular_license[host_is_superhost] = TRUE
)
```

```dax
Superhost Percentage = 
DIVIDE(
    [Superhost Count],
    CALCULATE(
        COUNTROWS(listings_regular_license),
        NOT(ISBLANK(listings_regular_license[host_is_superhost]))
    ),
    0
)
```

**Superhost Premium Individual** — Revenue multiplier for Superhost status (individual hosts)

```dax
Superhost Premium Individual = 
VAR SuperhostRevenue = 
    CALCULATE(
        AVERAGE(listings_regular_license[estimated_revenue_l365d]),
        listings_regular_license[host_category] = "Individual (1)",
        listings_regular_license[host_is_superhost] = TRUE
    )
VAR NonSuperhostRevenue = 
    CALCULATE(
        AVERAGE(listings_regular_license[estimated_revenue_l365d]),
        listings_regular_license[host_category] = "Individual (1)",
        listings_regular_license[host_is_superhost] = FALSE
    )
RETURN 
    DIVIDE(SuperhostRevenue, NonSuperhostRevenue, 0)
```

```dax
Superhost Premium Text = 
FORMAT([Superhost Premium Individual], "0.0") & "x"
```

**Sweet Spot Superhost Rate** — Superhost rate for small multi-property hosts (2-3 listings)

```dax
Sweet Spot Superhost Rate = 
CALCULATE(
    DIVIDE(
        CALCULATE(
            COUNTROWS(listings_regular_license),
            listings_regular_license[host_is_superhost] = TRUE
        ),
        CALCULATE(
            COUNTROWS(listings_regular_license),
            NOT(ISBLANK(listings_regular_license[host_is_superhost]))
        )
    ),
    listings_regular_license[host_category] = "Small Multi (2-3)"
)
```

**Medium Multi Superhost Downtown %** — Superhost rate for medium operators in downtown

```dax
Medium Multi Superhost Downtown Percentage = 
CALCULATE(
    DIVIDE(
        CALCULATE(
            COUNTROWS(listings_regular_license),
            listings_regular_license[host_category] = "Medium Multi (4-10)",
            listings_regular_license[host_is_superhost] = TRUE
        ),
        CALCULATE(
            COUNTROWS(listings_regular_license),
            listings_regular_license[host_category] = "Medium Multi (4-10)",
            NOT(ISBLANK(listings_regular_license[host_is_superhost]))
        )
    ),
    listings_regular_license[distance_cat] = "Downtown (<1km)"
)
```

### 2.5 Quality Metrics

**Large Multi Non-Superhost Rating** — Average rating for large operators without Superhost status

```dax
Large Multi Non-SH Rating = 
CALCULATE(
    AVERAGE(listings_regular_license[review_scores_rating]),
    listings_regular_license[host_category] = "Large Multi (11+)",
    listings_regular_license[host_is_superhost] = FALSE,
    NOT(ISBLANK(listings_regular_license[host_is_superhost]))
)
```

```dax
Large Multi Non-SH Rating Text = 
FORMAT([Large Multi Non-SH Rating], "0.00") & "⋆"
```

**Premium New Quality** — Average rating for new premium listings (>120€)

```dax
Premium New Quality = 
CALCULATE(
    AVERAGE(listings_regular_license[review_scores_rating]),
    YEAR(listings_regular_license[first_review_date]) = 2025,
    listings_regular_license[price_cat] = "Very High (>120€)"
)
```

```dax
Premium New Quality Text = 
FORMAT([Premium New Quality], "0.0") & "⋆"
```

**Rating Variability Ratio** — Quality consistency comparison (new vs mature listings)

```dax
Stdev Rise Ratio = 
VAR NewStdDev = 
    CALCULATE(
        STDEV.P(listings_regular_license[review_scores_rating]),
        listings_regular_license[market_maturity] = "New (<2yr)"
    )
VAR EstablishedStdDev = 
    CALCULATE(
        STDEV.P(listings_regular_license[review_scores_rating]),
        listings_regular_license[market_maturity] = "Mature (4-8yr)"
    )
RETURN
    DIVIDE(NewStdDev, EstablishedStdDev)
```

```dax
Stdev Rise Ratio Text = 
FORMAT([Stdev Rise Ratio], "0.0") & "x"
```

### 2.6 Market Maturity Metrics

**Median Listing Age**

```dax
Median Listing Age = 
MEDIAN(listings_regular_license[listing_age_years])
```

```dax
Median Listing Age Text = 
FORMAT([Median Listing Age], "0.00") & " yrs"
```

**New Listings Share** — Proportion of listings under 2 years old

```dax
New Listings Share = 
DIVIDE(
    CALCULATE(
        COUNT(listings_regular_license[id]),
        listings_regular_license[market_maturity] = "New (<2yr)"
    ),
    CALCULATE(
        COUNT(listings_regular_license[id]),
        REMOVEFILTERS(listings_regular_license[market_maturity])
    )
)
```

### 2.7 Price Distribution

**Price Category %** — Distribution across price tiers

```dax
Price_cat_% = 
VAR CurrentCount = COUNT(listings_regular_license[id])
VAR TotalCount = 
    CALCULATE(
        COUNT(listings_regular_license[id]),
        ALL(listings_regular_license[price_cat], listings_regular_license[price_sort]),
        listings_regular_license[price_cat] <> BLANK()
    )
RETURN
    DIVIDE(CurrentCount, TotalCount)
```

### 2.8 Calculated Columns

*Sorting columns for proper chart ordering in Power BI visuals.*

**Age Sort** — Numeric sort order for market maturity categories

```dax
age_sort = 
SWITCH(
    TRUE,
    ISBLANK(listings_regular_license[listing_age_years]), BLANK(),
    listings_regular_license[listing_age_years] <= 2, 1,
    listings_regular_license[listing_age_years] <= 4, 2,
    listings_regular_license[listing_age_years] <= 8, 3,
    4
)
```

**Distance Sort** — Numeric sort order for distance categories

```dax
dist_sort = 
SWITCH(
    TRUE,
    listings_regular_license[distance_to_center_km] < 1, 1,
    listings_regular_license[distance_to_center_km] < 3, 2,
    listings_regular_license[distance_to_center_km] < 6, 3,
    4
)
```

**Host Sort** — Numeric sort order for host categories

```dax
host_sort = 
SWITCH(
    TRUE,
    listings_regular_license[host_total_listings_count] = 1, 1,
    listings_regular_license[host_total_listings_count] < 4, 2,
    listings_regular_license[host_total_listings_count] < 11, 3,
    4
)
```

**Price Sort** — Numeric sort order for price categories

```dax
price_sort = 
SWITCH(
    TRUE,
    ISBLANK(listings_regular_license[price]), 0,
    listings_regular_license[price] <= 40, 1,
    listings_regular_license[price] <= 60, 2,
    listings_regular_license[price] <= 80, 3,
    listings_regular_license[price] <= 120, 4,
    5
)
```

---

## 3. Non-Regular License Listings

*Table: `listings_non_regular_license` — Contains listings without valid unique licenses (exempt, duplicated, or missing).*

### 3.1 License Category Counts

**Exempt Listings** — Properties exempt from licensing requirements

```dax
Exempt Count = 
CALCULATE(
    COUNTROWS(listings_cleaned),
    listings_cleaned[license] = "exempt"
)
```

```dax
Exempt Percentage = 
DIVIDE([Exempt Count], COUNTROWS(listings_cleaned), 0)
```

**Duplicate License Count** — Listings sharing a license number with others

```dax
Duplicate Count = 
COUNTROWS(listings_non_regular_license) - [Exempt Count] - [NA Count]
```

```dax
Duplicate Percentage = 
DIVIDE([Duplicate Count], COUNTROWS(listings_cleaned), 0)
```

**Missing License Count** — Listings with no license information

```dax
NA Count = 
CALCULATE(
    COUNTROWS(listings_cleaned),
    listings_cleaned[license] = BLANK()
)
```

```dax
NA Percentage = 
DIVIDE([NA Count], COUNTROWS(listings_cleaned), 0)
```

### 3.2 Duplicate Analysis Metrics

**Host Count** — Distinct hosts with duplicated licenses

```dax
host_count = 
CALCULATE(
    DISTINCTCOUNT(listings_non_regular_license[host_id]),
    listings_non_regular_license[License_Cat] = "Duplicated"
)
```

**Listings Count** — Total listings with duplicated licenses

```dax
listings_count = 
CALCULATE(
    COUNTROWS(listings_non_regular_license),
    listings_non_regular_license[License_Cat] = "Duplicated"
)
```

**Location Count** — Distinct locations with duplicated licenses

```dax
location_count = 
CALCULATE(
    DISTINCTCOUNT(listings_non_regular_license[location]),
    listings_non_regular_license[License_Cat] = "Duplicated"
)
```

**Top 2 Host Share** — Market concentration of top 2 hosts (by exempt listings)

```dax
Top 2 Host Share = 
VAR HostCounts = 
    SUMMARIZE(
        listings_non_regular_license, 
        listings_non_regular_license[host_id], 
        "exempt_count", COUNTROWS()
    )
VAR Top2Hosts = 
    TOPN(2, HostCounts, [exempt_count], DESC)
VAR numerator = 
    SUMX(Top2Hosts, [exempt_count])
VAR denominator = 
    COUNTROWS(listings_non_regular_license)
RETURN 
    DIVIDE(numerator, denominator)
```

### 3.3 Calculated Columns

**License Category** — Categorizes listings by license status

```dax
License_Cat = 
SWITCH(
    listings_non_regular_license[license], 
    "exempt", "Exempt", 
    BLANK(), "NA", 
    "Duplicated"
)
```

**Location Key** — Unique location identifier for proximity analysis

```dax
location = 
ROUND(listings_non_regular_license[latitude], 3) & "_" & 
ROUND(listings_non_regular_license[longitude], 3)
```

---

## Notes

- All measures handle null values appropriately using `DIVIDE()` with fallback values
- Sorting columns enable logical ordering in bar charts and slicers
- Text formatting measures append symbols (⋆, x) for dashboard KPI cards