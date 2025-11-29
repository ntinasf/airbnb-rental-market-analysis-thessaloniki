# Data Preprocessing Methodology

## Overview
This script processes raw Airbnb listings data for Thessaloniki, applying anonymization, cleaning, and validation before analysis.

## Pipeline Steps

### 1. Column Removal
Drops 34 columns containing:
- URLs and images
- Scraped metadata
- Text descriptions
- Redundant calculated fields

### 2. Anonymization
| Field | Method | Output Format |
|-------|--------|---------------|
| Listing ID | MD5 hash | `PROP_XXXXXX` |
| Host ID | MD5 hash | `HOST_XXXXXX` |
| License | MD5 hash / preserved | `LIC_XXXXXXXX` / `exempt` / `NaN` |
| Name | Property type + room type + sequence | `Ab - Apar - En #0001` |
| Coordinates | Rounding to 4 decimals (~11m accuracy) | Preserved for analysis |

### 3. Data Validation & Cleaning
Removes problematic entries:
- Top 10 extreme `minimum_nights` outliers
- Top 5 high price outliers
- Inactive listings (missing price/bathrooms/beds + zero reviews)

### 4. Type Conversions
- Percentages (`host_acceptance_rate`): string → float (0-1)
- Booleans (`host_is_superhost`, `instant_bookable`): 't'/'f' → True/False
- Price: `$1,234.00` → 1234.0
- Dates: string → datetime
- Categories: `neighbourhood_cleansed`, `property_type`, `room_type`

### 5. Feature Engineering
- **Host_Category**: Bins hosts by listing count
  - Individual (1)
  - Small Multi (2-3)
  - Large Multi (4+)

## Output
- Format: Parquet (compressed, efficient for analysis)
- Location: `data/processed/listings_cleaned.parquet`
- Includes processing report with validation stats
