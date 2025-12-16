from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

# Define anonymization functions


# Anonymization function for listing IDs
def anonymize_listing_id(series):
    """
    Anonymize listing IDs using deterministic hashing.

    Preserves: Uniqueness, consistency for joins
    Removes: Searchability, reverse-lookup capability

    Args:
        series (pd.Series): Original listing ID column

    Returns:
        pd.Series: Anonymized IDs in format 'PROP_XXXX'

    Example:
        12345678 -> PROP_A7B3
        98765432 -> PROP_C2F9
    """

    def hash_id(listing_id):
        if pd.isna(listing_id):
            return None
        # Create deterministic hash (same ID always gets same result)
        hash_obj = hashlib.md5(str(int(listing_id)).encode())
        # Take first 6 characters of hex digest for readability
        short_hash = hash_obj.hexdigest()[:6].upper()
        return f"PROP_{short_hash}"

    return series.apply(hash_id)


# Anonymization function for host IDs
def anonymize_host_id(series):
    """
    Anonymize host IDs using deterministic hashing.

    Preserves: Uniqueness, consistency for joins, host grouping
    Removes: Searchability, reverse-lookup capability

    Args:
        series (pd.Series): Original host ID column

    Returns:
        pd.Series: Anonymized IDs in format 'HOST_XXXX'

    Example:
        12345678 -> HOST_A7B3
        98765432 -> HOST_C2F9
    """

    def hash_id(host_id):
        if pd.isna(host_id):
            return None
        hash_obj = hashlib.md5(str(int(host_id)).encode())
        short_hash = hash_obj.hexdigest()[:6].upper()
        return f"HOST_{short_hash}"

    return series.apply(hash_id)


# Anonymization function for listing names
def anonymize_listing_name(
    df, name_col="name", property_type_col="property_type", room_type_col="room_type"
):
    """
    Generate anonymous but descriptive listing names.

    Preserves: Property character for analysis/storytelling
    Removes: Personal identifiers, searchable text

    Args:
        df (pd.DataFrame): Dataset with listing info
        name_col: Column name for listing names
        property_type_col: Column for property type (e.g., 'Apartment', 'House')
        room_type_col: Column for room type (e.g., 'Entire home/apt')

    Returns:
        pd.Series: Anonymized names in format 'Apartment - Entire Home #0042'
    """

    # Clean property type for missing values
    property_clean = df[property_type_col].fillna("Property")
    room_clean = df[room_type_col].fillna("Listing")

    # Create base descriptor combining property and room type
    base_descriptor = (
        df[name_col].str[:2]
        + " - "
        + property_clean.str[:4]
        + " - "
        + room_clean.str[:2]
    )

    # Add sequential number within each category for uniqueness
    sequential_num = df.groupby([property_type_col, room_type_col]).cumcount() + 1

    # Format with zero-padding for sorting
    anonymous_names = base_descriptor + " #" + sequential_num.astype(str).str.zfill(4)

    return anonymous_names


# Coordinates rounding function
def anonymize_coordinates(df, lat_col="latitude", lon_col="longitude", precision=3):
    """
    Reduce coordinate precision to prevent property identification.

    Preserves: Neighborhood clustering, relative distances, geographic patterns
    Removes: Exact property location, building-level identification

    Args:
        df (pd.DataFrame): Dataset with coordinates
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        precision: Decimal places to keep (default=3 for ~111m accuracy)

    Returns:
        tuple: (anonymized_lat, anonymized_lon) as pd.Series

    Precision levels:
        2 decimals = ~1.1 km
        3 decimals = ~111 m
        4 decimals = ~11 m
    """

    lat_anon = df[lat_col].round(precision)
    lon_anon = df[lon_col].round(precision)

    # Validation: Check for NaN values
    if lat_anon.isna().any() or lon_anon.isna().any():
        missing_count = lat_anon.isna().sum()
        print(f"⚠️  Warning: {missing_count} listings have missing coordinates")

    # Validation: Check coordinate ranges for Thessaloniki
    # Thessaloniki: ~40.6°N, ~22.9°E
    lat_valid = (lat_anon >= 40.4) & (lat_anon <= 40.8)
    lon_valid = (lon_anon >= 22.7) & (lon_anon <= 23.1)

    invalid_count = (~(lat_valid & lon_valid) & lat_anon.notna()).sum()
    if invalid_count > 0:
        print(
            f"⚠️  Warning: {invalid_count} listings have coordinates outside Thessaloniki bounds"
        )

    return lat_anon, lon_anon


# License anonymization function
def anonymize_license(series):
    """
    Anonymize license numbers while preserving licensing status categories.

    Preserves:
        - Licensed (numbered licenses -> hashed)
        - Exempt (remains 'exempt')
        - Unlicensed (null/empty -> remains null)
        - Duplicate licenses (same number -> same hash)

    Removes: Actual license numbers, registry searchability

    Args:
        series (pd.Series): Original license column

    Returns:
        pd.Series: Anonymized licenses with three categories:
            - 'LIC_XXXXXX' for numbered licenses
            - 'exempt' for exempt properties
            - NaN for unlicensed properties

    Example:
        '123456789'  -> 'LIC_A7B3C2'
        '123456789'  -> 'LIC_A7B3C2'  (duplicate preserved)
        '987654321'  -> 'LIC_F9E8D1'  (different hash)
        'exempt'     -> 'exempt'
        NaN          -> NaN
        ''           -> NaN
    """

    def hash_license(value):
        # Handle missing/empty values -> Unlicensed
        if pd.isna(value) or value == "":
            return None

        # Convert to string and strip whitespace
        value_str = str(value).strip().lower()

        # Preserve 'exempt' status as-is
        if value_str == "exempt":
            return "exempt"

        # For numbered licenses, create deterministic hash
        # MD5 ensures same license always gets same hash (duplicates preserved)
        hash_obj = hashlib.md5(str(value).encode())
        short_hash = hash_obj.hexdigest()[:8].upper()
        return f"LIC_{short_hash}"

    anonymized = series.apply(hash_license)

    return anonymized


# Reporting function
def print_processing_report(original_df, processed_df, columns_dropped, stats):
    """Print concise report of data transformations."""
    print("\n" + "=" * 60)
    print("DATA PROCESSING REPORT")
    print("=" * 60)

    print("\n📊 Dataset Overview:")
    print(f"   Rows: {len(original_df):,} → {len(processed_df):,}")
    print(f"   Columns: {len(original_df.columns)} → {len(processed_df.columns)}")
    print(f"   Dropped: {len(columns_dropped)} columns")

    print("\n🧹 Data Validation:")
    print(f"   Problematic entries removed: {stats['removed_entries']:,}")
    if stats["removed_entries"] > 0:
        print(f"      - Extreme minimum_nights: {stats['extreme_min_nights']:,}")
        print(f"      - High outlier prices: {stats['high_price_outliers']:,}")
        print(f"      - Incomplete/inactive: {stats['incomplete_inactive']:,}")
        print(
            f"      - Dead listings (zero activity + no reviews): {stats['dead_listings']:,}"
        )

    print("\n� Missing Value Imputation:")
    print(f"   host_category imputed: {stats['host_category_imputed']:,} listings")
    print(f"   Unique hosts affected: {stats['hosts_imputed']:,}")

    print("\n�🔒 Anonymization:")
    print(f"   Listing IDs: {stats['ids_anonymized']:,}")
    print(f"   Host IDs: {stats['host_ids_anonymized']:,}")
    print(f"   Names: {stats['names_anonymized']:,}")
    print(f"   Licenses: {stats['licenses_anonymized']:,}")
    print(f"   Coordinates: rounded to {stats['coord_precision']} decimals")

    print("\n🔍 Distinct Values (Before → After Anonymization):")
    distinct_before = stats["distinct_before"]
    distinct_after = stats["distinct_after"]

    def check_integrity(before, after, name):
        status = "✓" if before == after else "⚠️"
        return f"   {status} {name}: {before:,} → {after:,}"

    print(
        check_integrity(
            distinct_before["listing_id"], distinct_after["listing_id"], "Listing IDs"
        )
    )
    print(
        check_integrity(
            distinct_before["host_id"], distinct_after["host_id"], "Host IDs"
        )
    )
    print(check_integrity(distinct_before["name"], distinct_after["name"], "Names"))
    print(
        check_integrity(
            distinct_before["license"], distinct_after["license"], "Licenses"
        )
    )
    print(
        check_integrity(
            distinct_before["coords"], distinct_after["coords"], "Coordinates"
        )
    )

    print("\n📋 License Distribution:")
    print(f"   Licensed: {stats['licensed']:,} ({stats['licensed_pct']:.1f}%)")
    print(f"   Exempt: {stats['exempt']:,} ({stats['exempt_pct']:.1f}%)")
    print(f"   Unlicensed: {stats['unlicensed']:,} ({stats['unlicensed_pct']:.1f}%)")
    if stats["duplicate_licenses"] > 0:
        print(f"   ⚠️  Duplicates: {stats['duplicate_licenses']:,}")

    print("\n✅ Data Quality:")
    print(f"   Missing coords: {stats['missing_coords']:,}")
    print(f"   Invalid coords: {stats['invalid_coords']:,}")

    memory_mb = processed_df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n💾 Output: {memory_mb:.2f} MB (Parquet)")
    print("=" * 60 + "\n")


# Execute anonymization and preprocessing pipeline
if __name__ == "__main__":

    raw_data_path = Path("data") / "raw" / "listings.csv.gz"
    processed_data_path = Path("data") / "processed" / "listings_cleaned.parquet"

    columns_to_drop = [
        "listing_url",
        "scrape_id",
        "last_scraped",
        "source",
        "description",
        "neighborhood_overview",
        "picture_url",
        "host_url",
        "host_name",
        "host_about",
        "host_thumbnail_url",
        "host_picture_url",
        "host_neighbourhood",
        "host_listings_count",
        "host_verifications",
        "host_has_profile_pic",
        "neighbourhood_group_cleansed",
        "bathrooms_text",
        "amenities",
        "minimum_minimum_nights",
        "maximum_minimum_nights",
        "minimum_maximum_nights",
        "maximum_maximum_nights",
        "minimum_nights_avg_ntm",
        "maximum_nights_avg_ntm",
        "calendar_last_scraped",
        "number_of_reviews_l30d",
        "calculated_host_listings_count",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "calendar_updated",
        "has_availability",
    ]

    # Load dataset
    original_listings = pd.read_csv(raw_data_path)
    listings = original_listings.drop(columns=columns_to_drop)

    # Track original license stats and distinct values before anonymization
    original_license = listings["license"].copy()

    # Count distinct values BEFORE anonymization
    distinct_before = {
        "listing_id": listings["id"].nunique(),
        "host_id": listings["host_id"].nunique(),
        "name": listings["name"].nunique(),
        "license": listings["license"].nunique(),
        "coords": listings[["latitude", "longitude"]].drop_duplicates().shape[0],
    }

    # Apply anonymization functions
    listings["id"] = anonymize_listing_id(listings["id"])
    listings["host_id"] = anonymize_host_id(listings["host_id"])
    listings["name"] = anonymize_listing_name(
        listings,
        name_col="name",
        property_type_col="property_type",
        room_type_col="room_type",
    )
    listings["latitude"], listings["longitude"] = anonymize_coordinates(
        listings, lat_col="latitude", lon_col="longitude", precision=4
    )
    listings["license"] = anonymize_license(listings["license"])

    # Count distinct values AFTER anonymization
    distinct_after = {
        "listing_id": listings["id"].nunique(),
        "host_id": listings["host_id"].nunique(),
        "name": listings["name"].nunique(),
        "license": listings["license"].nunique(),
        "coords": listings[["latitude", "longitude"]].drop_duplicates().shape[0],
    }

    # Data type conversions and cleaning
    listings["host_acceptance_rate"] = (
        listings["host_acceptance_rate"].str.rstrip("%").astype(float) / 100.0
    )
    listings["host_is_superhost"] = listings["host_is_superhost"].map(
        {"t": True, "f": False}
    )
    listings["host_identity_verified"] = listings["host_identity_verified"].map(
        {"t": True, "f": False}
    )
    listings["price"] = (
        listings["price"].str.lstrip("$").str.replace(",", "").astype(float)
    )
    listings["instant_bookable"] = listings["instant_bookable"].map(
        {"t": True, "f": False}
    )

    type_dict = {
        "id": "object",
        "host_id": "object",
        "host_since": "datetime64[ns]",
        "host_total_listings_count": "Int16",
        "neighbourhood_cleansed": "category",
        "property_type": "category",
        "room_type": "category",
        "accommodates": "Int16",
        "bedrooms": "Int16",
        "beds": "Int16",
        "minimum_nights": "Int16",
        "maximum_nights": "Int16",
    }

    listings = listings.astype(type_dict)

    # ========== DATA VALIDATION: Remove problematic entries ==========

    # Track removed entries for reporting
    initial_count = len(listings)

    # 1. Top 10 extreme minimum_nights outliers
    indices_1 = (
        listings.sort_values(by="minimum_nights", ascending=False).head(10).index
    )

    # 2. Top 4 high price problematic outliers
    indices_2 = listings.sort_values(by="price", ascending=False).head(5).index

    # 3. Incomplete/inactive listings (missing critical data + no reviews)
    indices_3 = listings[
        listings["price"].isna()
        & listings["bathrooms"].isna()
        & listings["beds"].isna()
        & (listings["number_of_reviews_ltm"] == 0)
        & (listings["number_of_reviews_ly"] == 0)
    ].index

    # 4. Dead/inactive listings (zero activity metrics + missing review data)
    # These are listings with no booking activity AND no review history
    indices_4 = listings[
        (listings["number_of_reviews"] == 0)
        & (listings["estimated_occupancy_l365d"] == 0)
        & (listings["estimated_revenue_l365d"] == 0)
        & listings["first_review"].isna()
        & listings["last_review"].isna()
        & listings["review_scores_rating"].isna()
    ].index

    # Combine all indices to remove (union prevents duplicates)
    indices_to_remove = indices_1.union(indices_2).union(indices_3).union(indices_4)

    # Track removal statistics
    removed_stats = {
        "extreme_min_nights": len(indices_1),
        "high_price_outliers": len(indices_2),
        "incomplete_inactive": len(indices_3),
        "dead_listings": len(indices_4),
        "total_removed": len(indices_to_remove),
    }

    # Initialize imputation tracking (will be updated later)
    imputed_count = 0
    imputed_hosts = 0

    # Remove problematic entries
    listings.drop(index=indices_to_remove, inplace=True)

    # ========== END DATA VALIDATION ==========

    # ========== FEATURE ENGINEERING ==========

    # Host categories based on listing count (4-tier system)
    def categorize_host_count(count):
        if pd.isna(count):
            return "Unknown"
        elif count == 1:
            return "Individual (1)"
        elif count <= 3:
            return "Small Multi (2-3)"
        elif count <= 10:
            return "Medium Multi (4-10)"
        else:
            return "Large Multi (11+)"

    listings["host_category"] = listings["host_total_listings_count"].apply(
        categorize_host_count
    )

    # ========== MISSING VALUE IMPUTATION: host_category ==========
    # Some listings have missing host_total_listings_count but we can infer
    # the count from how many listings each host has in our dataset

    unknown_mask = listings["host_category"] == "Unknown"
    unknown_host_ids = listings.loc[unknown_mask, "host_id"].unique()

    if len(unknown_host_ids) > 0:
        # Count actual listings per host in our dataset
        host_counts = listings.loc[
            listings["host_id"].isin(unknown_host_ids), "host_id"
        ].value_counts()

        # Create category mapping from actual counts
        host_category_map = host_counts.apply(categorize_host_count)

        # Impute host_total_listings_count with actual counts
        host_listings_count_map = host_counts.to_dict()
        listings.loc[unknown_mask, "host_total_listings_count"] = listings.loc[
            unknown_mask, "host_id"
        ].map(host_listings_count_map)

        # Update host_category with imputed values
        listings["host_category"] = (
            listings["host_id"].map(host_category_map).fillna(listings["host_category"])
        )

        imputed_count = unknown_mask.sum()
        imputed_hosts = len(unknown_host_ids)
    else:
        imputed_count = 0
        imputed_hosts = 0

    # ========== END MISSING VALUE IMPUTATION ==========

    # Geographic features: Distance from city center
    from geopy.distance import geodesic

    landmark_coords = (
        40.62962,
        22.94473,
    )  # Midpoint: White Tower / Aristotelous Square

    listings["distance_to_center_km"] = listings.apply(
        lambda row: (
            geodesic((row["latitude"], row["longitude"]), landmark_coords).km
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
            else np.nan
        ),
        axis=1,
    )

    listings["distance_cat"] = pd.cut(
        listings["distance_to_center_km"],
        bins=[0, 1, 3, 6, 100],
        labels=[
            "Downtown (<1km)",
            "Inner City (1-3km)",
            "Neighborhoods (3-6km)",
            "Suburban (>6km)",
        ],
    )

    # Price categories
    listings["price_cat"] = pd.cut(
        listings["price"],
        bins=[0, 40, 60, 80, 120, np.inf],
        labels=[
            "Very Low (<40€)",
            "Low (40-60€)",
            "Medium (60-80€)",
            "High (80-120€)",
            "Very High (>120€)",
        ],
    )

    # Temporal features: Listing age and market maturity
    listings["first_review_date"] = pd.to_datetime(listings["first_review"])
    listings["last_review_date"] = pd.to_datetime(listings["last_review"])

    reference_date = listings["last_review_date"].max()

    listings["listing_age_years"] = (
        reference_date - listings["first_review_date"]
    ).dt.days / 365.25

    listings["market_maturity"] = pd.cut(
        listings["listing_age_years"],
        bins=[-1, 2, 4, 8, 100],
        labels=[
            "New (<2yr)",
            "Growing (2-4yr)",
            "Mature (4-8yr)",
            "Established (>8yr)",
        ],
    )

    # ========== END FEATURE ENGINEERING ==========

    # Collect statistics for report
    total = len(listings)
    licensed = (
        listings["license"].notna()
        & (listings["license"] != "exempt")
        & (listings["license"].str.startswith("LIC_"))
    ).sum()
    exempt = (listings["license"] == "exempt").sum()
    unlicensed = listings["license"].isna().sum()

    # Count duplicate licenses in original data
    original_lic_clean = original_license[
        original_license.notna() & (original_license != "exempt")
    ]
    duplicate_licenses = (original_lic_clean.value_counts() > 1).sum()

    # Coordinate stats
    missing_coords = listings["latitude"].isna().sum()
    lat_valid = (listings["latitude"] >= 40.4) & (listings["latitude"] <= 40.8)
    lon_valid = (listings["longitude"] >= 22.7) & (listings["longitude"] <= 23.1)
    invalid_coords = (~(lat_valid & lon_valid) & listings["latitude"].notna()).sum()

    stats = {
        "removed_entries": removed_stats["total_removed"],
        "extreme_min_nights": removed_stats["extreme_min_nights"],
        "high_price_outliers": removed_stats["high_price_outliers"],
        "incomplete_inactive": removed_stats["incomplete_inactive"],
        "dead_listings": removed_stats["dead_listings"],
        "host_category_imputed": imputed_count,
        "hosts_imputed": imputed_hosts,
        "ids_anonymized": listings["id"].notna().sum(),
        "host_ids_anonymized": listings["host_id"].notna().sum(),
        "names_anonymized": listings["name"].notna().sum(),
        "licenses_anonymized": licensed,
        "coord_precision": 4,
        "licensed": licensed,
        "licensed_pct": (licensed / total) * 100,
        "exempt": exempt,
        "exempt_pct": (exempt / total) * 100,
        "unlicensed": unlicensed,
        "unlicensed_pct": (unlicensed / total) * 100,
        "duplicate_licenses": duplicate_licenses,
        "missing_coords": missing_coords,
        "invalid_coords": invalid_coords,
        "distinct_before": distinct_before,
        "distinct_after": distinct_after,
    }

    # Save processed dataset
    listings.to_parquet(processed_data_path, engine="pyarrow", index=False)

    # Print report
    print_processing_report(original_listings, listings, columns_to_drop, stats)
    print(f"✅ Saved to: {processed_data_path}")
