# on listings cleaned

Total Listings (All Data) = 
COUNTROWS(listings_cleaned)

Compliance Rate = 
DIVIDE([Total Listings], [Total Listings (All Data)], 0)

# on geographic temporal

Total Listings = 
COUNTROWS(listings_geographic_temporal)

// REVENUE & PRICE METRICS
// ============================================

Avg Revenue = 
AVERAGE(listings_geographic_temporal[estimated_revenue_l365d])

Median Revenue = 
MEDIAN(listings_geographic_temporal[estimated_revenue_l365d])

Avg Price = 
AVERAGE(listings_geographic_temporal[price])

Median Price = 
MEDIAN(listings_geographic_temporal[price])

/ HOST METRICS
// ============================================

Superhost Count = 
CALCULATE(
    COUNTROWS(listings_geographic_temporal),
    listings_geographic_temporal[is_superhost] = TRUE
)

Superhost Percentage = 
DIVIDE(
    [Superhost Count],
    [Total Listings],
    0
)

Multi-Host Count = 
CALCULATE(
    COUNTROWS(listings_geographic_temporal),
    listings_geographic_temporal[Host_Category] = "Large Multi (4+)"
)

Multi-Host Percentage = 
DIVIDE(
    [Multi-Host Count],
    [Total Listings],
    0
)

Superhost Premium Individual = 
VAR SuperhostRevenue = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[estimated_revenue_l365d]),
        listings_geographic_temporal[Host_Category] = "Individual (1)",
        listings_geographic_temporal[is_superhost] = TRUE
    )
VAR NonSuperhostRevenue = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[estimated_revenue_l365d]),
        listings_geographic_temporal[Host_Category] = "Individual (1)",
        listings_geographic_temporal[is_superhost] = FALSE
    )
RETURN 
    DIVIDE(SuperhostRevenue, NonSuperhostRevenue, 0)

Superhost Premium Text = 
FORMAT([Superhost Premium Individual], "0.0") & "x"

Quality Decline Percentage = 
VAR Pre2022Rating = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[review_scores_rating]),
        listings_geographic_temporal[Host_Category] = "Large Multi (4+)",
        YEAR(listings_geographic_temporal[first_review_date]) <= 2022
    )
VAR Post2023Rating = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[review_scores_rating]),
        listings_geographic_temporal[Host_Category] = "Large Multi (4+)",
        YEAR(listings_geographic_temporal[first_review_date]) >= 2023
    )
VAR Decline = Post2023Rating - Pre2022Rating
RETURN 
    DIVIDE(Decline, Pre2022Rating, 0)

Listings by Neighborhood = 
CALCULATE(
    COUNTROWS(listings_geographic_temporal),
    ALLEXCEPT(listings_geographic_temporal, listings_geographic_temporal[neighbourhood_cleansed])
)

Avg Price by Neighborhood = 
CALCULATE(
    AVERAGE(listings_geographic_temporal[price]),
    ALLEXCEPT(listings_geographic_temporal, listings_geographic_temporal[neighbourhood_cleansed])
)

Sweet Spot Superhost Rate = 
CALCULATE(
    DIVIDE(
        CALCULATE(
            COUNTROWS(listings_geographic_temporal),
            listings_geographic_temporal[host_is_superhost] = TRUE
        ),
        COUNTROWS(listings_geographic_temporal)
    ),
    listings_geographic_temporal[Host_Category] = "Small Multi (2-3)"
)

Large Multi Non-SH Rating = 
CALCULATE(
    AVERAGE(listings_geographic_temporal[review_scores_rating]),
    listings_geographic_temporal[Host_Category] = "Large Multi (4+)",
    listings_geographic_temporal[host_is_superhost] = FALSE
)

Revenue Multiplier Small Multi = 
VAR SH = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[estimated_revenue_l365d]),
        listings_geographic_temporal[Host_Category] = "Small Multi (2-3)",
        listings_geographic_temporal[host_is_superhost] = TRUE
    )
VAR NonSH = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[estimated_revenue_l365d]),
        listings_geographic_temporal[Host_Category] = "Small Multi (2-3)",
        listings_geographic_temporal[host_is_superhost] = FALSE
    )
RETURN DIVIDE(SH, NonSH, 0)

Revenue Multiplier Large Multi = 
VAR SH = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[estimated_revenue_l365d]),
        listings_geographic_temporal[Host_Category] = "Large Multi (4+)",
        listings_geographic_temporal[host_is_superhost] = TRUE
    )
VAR NonSH = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[estimated_revenue_l365d]),
        listings_geographic_temporal[Host_Category] = "Large Multi (4+)",
        listings_geographic_temporal[host_is_superhost] = FALSE
    )
RETURN DIVIDE(SH, NonSH, 0)

Multi-Host Downtown Percentage = 
CALCULATE(
    DIVIDE(
        CALCULATE(
            COUNTROWS(listings_geographic_temporal),
            listings_geographic_temporal[Host_Category] = "Large Multi (4+)"
        ),
        COUNTROWS(listings_geographic_temporal)
    ),
    listings_geographic_temporal[distance_cat] = "Downtown (<1km)"
)

Multi-Host Neighborhoods Percentage = 
CALCULATE(
    DIVIDE(
        CALCULATE(
            COUNTROWS(listings_geographic_temporal),
            listings_geographic_temporal[Host_Category] = "Large Multi (4+)"
        ),
        COUNTROWS(listings_geographic_temporal)
    ),
    listings_geographic_temporal[distance_cat] = "Neighborhoods (3-6km)"
)

Host Category Distribution = 
VAR CurrentLocation = SELECTEDVALUE(listings_geographic_temporal[distance_cat])
VAR CurrentHostCat = SELECTEDVALUE(listings_geographic_temporal[Host_Category])
VAR CountInCategory = 
    CALCULATE(
        COUNTROWS(listings_geographic_temporal),
        listings_geographic_temporal[distance_cat] = CurrentLocation,
        listings_geographic_temporal[Host_Category] = CurrentHostCat
    )
VAR TotalInLocation = 
    CALCULATE(
        COUNTROWS(listings_geographic_temporal),
        listings_geographic_temporal[distance_cat] = CurrentLocation
    )
RETURN
    DIVIDE(CountInCategory, TotalInLocation, 0)

Median Listing Age = 
MEDIAN(listings_geographic_temporal[listing_age_years])

Budget Segment Quality Decline = 
VAR Pre2022 = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[review_scores_rating]),
        listings_geographic_temporal[Host_Category] = "Large Multi (4+)",
        listings_geographic_temporal[price_cat] = "Very Low (<40€)",
        YEAR(listings_geographic_temporal[first_review]) <= 2022
    )
VAR Post2023 = 
    CALCULATE(
        AVERAGE(listings_geographic_temporal[review_scores_rating]),
        listings_geographic_temporal[Host_Category] = "Large Multi (4+)",
        listings_geographic_temporal[price_cat] = "Very Low (<40€)",
        YEAR(listings_geographic_temporal[first_review]) >= 2023
    )
RETURN DIVIDE(Post2023 - Pre2022, Pre2022, 0)

Avg Rating by Cohort = 
VAR CurrentYear = SELECTEDVALUE(listings_geographic_temporal[first_review_date])
VAR CurrentHostCat = SELECTEDVALUE(listings_geographic_temporal[Host_Category])
VAR CurrentPriceCat = SELECTEDVALUE(listings_geographic_temporal[price_cat])
RETURN
    CALCULATE(
        AVERAGE(listings_geographic_temporal[review_scores_rating]),
        YEAR(listings_geographic_temporal[first_review_date]) = CurrentYear,
        listings_geographic_temporal[Host_Category] = CurrentHostCat,
        listings_geographic_temporal[price_cat] = CurrentPriceCat
    )




# on non regular

Exempt Count = 
CALCULATE(
    COUNTROWS(listings_cleaned),
    listings_cleaned[license] = "exempt"
)

Exempt Percentage = 
DIVIDE([Exempt Count], COUNTROWS(listings_cleaned), 0)

NA Count = 
CALCULATE(
    COUNTROWS(listings_cleaned),
    listings_cleaned[license] = BLANK()
)

NA Percentage = 
DIVIDE([NA Count], COUNTROWS(listings_cleaned), 0)

Duplicate Count = 
COUNTROWS(listings_non_regular_license) - [Exempt Count] - [NA Count]

Duplicate Percentage = 
DIVIDE([Duplicate Count], COUNTROWS(listings_cleaned), 0)

Properties per Host (Exempt) = 
CALCULATE(
    COUNTROWS(listings_cleaned),
    listings_cleaned[license] = "exempt"
) / 
DISTINCTCOUNT(listings_cleaned[host_id])
