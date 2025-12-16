# Power BI DAX Formulas

This document contains all DAX measures used in the Airbnb Thessaloniki analysis dashboard.

---

## Table: `listings_cleaned`

### Total Listings (All Data)

```dax
Total Listings (All Data) = 
COUNTROWS(listings_cleaned)
```

### Compliance Rate

```dax
Compliance Rate = 
DIVIDE([Total Listings], [Total Listings (All Data)], 0)
```

---

## Table: `listings_regular_license`

### Basic Counts

#### Total Listings

```dax
Total Listings = 
COUNTROWS(listings_regular_license)
```

---

### Revenue & Price Metrics

#### Avg Revenue

```dax
Avg Revenue = 
AVERAGE(listings_regular_license[estimated_revenue_l365d])
```

#### Median Revenue

```dax
Median Revenue = 
MEDIAN(listings_regular_license[estimated_revenue_l365d])
```

#### Avg Price

```dax
Avg Price = 
AVERAGE(listings_regular_license[price])
```

#### Median Price

```dax
Median Price = 
MEDIAN(listings_regular_license[price])
```

---

### Host Metrics

#### Superhost Count

```dax
Superhost Count = 
CALCULATE(
    COUNTROWS(listings_regular_license),
    listings_regular_license[is_superhost] = TRUE
)
```

#### Superhost Percentage

```dax
Superhost Percentage = 
DIVIDE(
    [Superhost Count],
    [Total Listings],
    0
)
```

#### Multi-Host Count

```dax
Multi-Host Count = 
CALCULATE(
    COUNTROWS(listings_regular_license),
    listings_regular_license[host_category] = "Large Multi (11+)"
)
```

#### Multi-Host Percentage

```dax
Multi-Host Percentage = 
DIVIDE(
    [Multi-Host Count],
    [Total Listings],
    0
)
```

#### Superhost Premium Individual

```dax
Superhost Premium Individual = 
VAR SuperhostRevenue = 
    CALCULATE(
        AVERAGE(listings_regular_license[estimated_revenue_l365d]),
        listings_regular_license[host_category] = "Individual (1)",
        listings_regular_license[is_superhost] = TRUE
    )
VAR NonSuperhostRevenue = 
    CALCULATE(
        AVERAGE(listings_regular_license[estimated_revenue_l365d]),
        listings_regular_license[host_category] = "Individual (1)",
        listings_regular_license[is_superhost] = FALSE
    )
RETURN 
    DIVIDE(SuperhostRevenue, NonSuperhostRevenue, 0)
```

#### Superhost Premium Text

```dax
Superhost Premium Text = 
FORMAT([Superhost Premium Individual], "0.0") & "x"
```

#### Sweet Spot Superhost Rate

```dax
Sweet Spot Superhost Rate = 
CALCULATE(
    DIVIDE(
        CALCULATE(
            COUNTROWS(listings_regular_license),
            listings_regular_license[host_is_superhost] = TRUE
        ),
        COUNTROWS(listings_regular_license)
    ),
    listings_regular_license[host_category] = "Small Multi (2-3)"
)
```

#### Large Multi Non-SH Rating

```dax
Large Multi Non-SH Rating = 
CALCULATE(
    AVERAGE(listings_regular_license[review_scores_rating]),
    listings_regular_license[host_category] = "Large Multi (11+)",
    listings_regular_license[host_is_superhost] = FALSE
)
```

---

### Revenue Multipliers

#### Revenue Multiplier Small Multi

```dax
Revenue Multiplier Small Multi = 
VAR SH = 
    CALCULATE(
        AVERAGE(listings_regular_license[estimated_revenue_l365d]),
        listings_regular_license[host_category] = "Small Multi (2-3)",
        listings_regular_license[host_is_superhost] = TRUE
    )
VAR NonSH = 
    CALCULATE(
        AVERAGE(listings_regular_license[estimated_revenue_l365d]),
        listings_regular_license[host_category] = "Small Multi (2-3)",
        listings_regular_license[host_is_superhost] = FALSE
    )
RETURN 
    DIVIDE(SH, NonSH, 0)
```

#### Revenue Multiplier Large Multi

```dax
Revenue Multiplier Large Multi = 
VAR SH = 
    CALCULATE(
        AVERAGE(listings_regular_license[estimated_revenue_l365d]),
        listings_regular_license[host_category] = "Large Multi (11+)",
        listings_regular_license[host_is_superhost] = TRUE
    )
VAR NonSH = 
    CALCULATE(
        AVERAGE(listings_regular_license[estimated_revenue_l365d]),
        listings_regular_license[host_category] = "Large Multi (11+)",
        listings_regular_license[host_is_superhost] = FALSE
    )
RETURN 
    DIVIDE(SH, NonSH, 0)
```

---

### Geographic Metrics

#### Listings by Neighborhood

```dax
Listings by Neighborhood = 
CALCULATE(
    COUNTROWS(listings_regular_license),
    ALLEXCEPT(listings_regular_license, listings_regular_license[neighbourhood_cleansed])
)
```

#### Avg Price by Neighborhood

```dax
Avg Price by Neighborhood = 
CALCULATE(
    AVERAGE(listings_regular_license[price]),
    ALLEXCEPT(listings_regular_license, listings_regular_license[neighbourhood_cleansed])
)
```

#### Multi-Host Downtown Percentage

```dax
Multi-Host Downtown Percentage = 
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

#### Multi-Host Neighborhoods Percentage

```dax
Multi-Host Neighborhoods Percentage = 
CALCULATE(
    DIVIDE(
        CALCULATE(
            COUNTROWS(listings_regular_license),
            listings_regular_license[host_category] = "Large Multi (11+)"
        ),
        COUNTROWS(listings_regular_license)
    ),
    listings_regular_license[distance_cat] = "Neighborhoods (3-6km)"
)
```

#### Host Category Distribution

```dax
Host Category Distribution = 
VAR CurrentLocation = SELECTEDVALUE(listings_regular_license[distance_cat])
VAR CurrentHostCat = SELECTEDVALUE(listings_regular_license[host_category])
VAR CountInCategory = 
    CALCULATE(
        COUNTROWS(listings_regular_license),
        listings_regular_license[distance_cat] = CurrentLocation,
        listings_regular_license[host_category] = CurrentHostCat
    )
VAR TotalInLocation = 
    CALCULATE(
        COUNTROWS(listings_regular_license),
        listings_regular_license[distance_cat] = CurrentLocation
    )
RETURN
    DIVIDE(CountInCategory, TotalInLocation, 0)
```

---

### Quality & Temporal Metrics

#### Median Listing Age

```dax
Median Listing Age = 
MEDIAN(listings_regular_license[listing_age_years])
```

#### Quality Decline Percentage

```dax
Quality Decline Percentage = 
VAR Pre2022Rating = 
    CALCULATE(
        AVERAGE(listings_regular_license[review_scores_rating]),
        listings_regular_license[host_category] = "Large Multi (11+)",
        YEAR(listings_regular_license[first_review_date]) <= 2022
    )
VAR Post2023Rating = 
    CALCULATE(
        AVERAGE(listings_regular_license[review_scores_rating]),
        listings_regular_license[host_category] = "Large Multi (11+)",
        YEAR(listings_regular_license[first_review_date]) >= 2023
    )
VAR Decline = Post2023Rating - Pre2022Rating
RETURN 
    DIVIDE(Decline, Pre2022Rating, 0)
```

#### Budget Segment Quality Decline

```dax
Budget Segment Quality Decline = 
VAR Pre2022 = 
    CALCULATE(
        AVERAGE(listings_regular_license[review_scores_rating]),
        listings_regular_license[host_category] = "Large Multi (11+)",
        listings_regular_license[price_cat] = "Very Low (<40€)",
        YEAR(listings_regular_license[first_review]) <= 2022
    )
VAR Post2023 = 
    CALCULATE(
        AVERAGE(listings_regular_license[review_scores_rating]),
        listings_regular_license[host_category] = "Large Multi (11+)",
        listings_regular_license[price_cat] = "Very Low (<40€)",
        YEAR(listings_regular_license[first_review]) >= 2023
    )
RETURN 
    DIVIDE(Post2023 - Pre2022, Pre2022, 0)
```

#### Avg Rating by Cohort

```dax
Avg Rating by Cohort = 
VAR CurrentYear = SELECTEDVALUE(listings_regular_license[first_review_date])
VAR CurrentHostCat = SELECTEDVALUE(listings_regular_license[host_category])
VAR CurrentPriceCat = SELECTEDVALUE(listings_regular_license[price_cat])
RETURN
    CALCULATE(
        AVERAGE(listings_regular_license[review_scores_rating]),
        YEAR(listings_regular_license[first_review_date]) = CurrentYear,
        listings_regular_license[host_category] = CurrentHostCat,
        listings_regular_license[price_cat] = CurrentPriceCat
    )
```

---

## Table: `listings_non_regular_license`

### Exempt Count

```dax
Exempt Count = 
CALCULATE(
    COUNTROWS(listings_cleaned),
    listings_cleaned[license] = "exempt"
)
```

### Exempt Percentage

```dax
Exempt Percentage = 
DIVIDE([Exempt Count], COUNTROWS(listings_cleaned), 0)
```

### NA Count

```dax
NA Count = 
CALCULATE(
    COUNTROWS(listings_cleaned),
    listings_cleaned[license] = BLANK()
)
```

### NA Percentage

```dax
NA Percentage = 
DIVIDE([NA Count], COUNTROWS(listings_cleaned), 0)
```

### Duplicate Count

```dax
Duplicate Count = 
COUNTROWS(listings_non_regular_license) - [Exempt Count] - [NA Count]
```

### Duplicate Percentage

```dax
Duplicate Percentage = 
DIVIDE([Duplicate Count], COUNTROWS(listings_cleaned), 0)
```

### Properties per Host (Exempt)

```dax
Properties per Host (Exempt) = 
CALCULATE(
    COUNTROWS(listings_cleaned),
    listings_cleaned[license] = "exempt"
) / 
DISTINCTCOUNT(listings_cleaned[host_id])
```
