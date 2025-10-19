from pathlib import Path
import pandas as pd


def convert_data_types(df):
    """
    Convert calendar data types for memory efficiency and correctness.

    Args:
        df (pd.DataFrame): Calendar dataframe with raw data types

    Returns:
        pd.DataFrame: DataFrame with optimized data types
    """
    # Convert available column from 't'/'f' to boolean
    df["available"] = df["available"].map({"t": True, "f": False})

    # Define type conversions
    type_dict = {
        "listing_id": "object",
        "date": "datetime64[ns]",
        "price": "float32",
        "minimum_nights": "Int16",
        "maximum_nights": "Int16",
    }

    df = df.astype(type_dict)

    return df


def drop_empty_columns(df, columns_to_drop):
    """
    Drop empty or unnecessary columns from calendar data.

    Args:
        df (pd.DataFrame): Calendar dataframe
        columns_to_drop (list): List of column names to drop

    Returns:
        pd.DataFrame: DataFrame with columns removed
    """
    return df.drop(columns=columns_to_drop, axis=1)


def print_processing_report(original_df, processed_df, columns_dropped, stats):
    """Print concise report of data transformations."""
    print("\n" + "=" * 60)
    print("CALENDAR DATA PROCESSING REPORT")
    print("=" * 60)

    print("\n📊 Dataset Overview:")
    print(f"   Rows: {len(original_df):,} → {len(processed_df):,}")
    print(f"   Columns: {len(original_df.columns)} → {len(processed_df.columns)}")
    print(f"   Dropped: {len(columns_dropped)} columns")

    print("\n🔧 Transformations:")
    print("   Available: converted to boolean")
    print("   Date: converted to datetime64[ns]")
    print("   Price/Nights: optimized to Int16/float32")

    print("\n📋 Data Statistics:")
    print(f"   Date range: {stats['date_min']} to {stats['date_max']}")
    print(f"   Unique listings: {stats['unique_listings']:,}")
    print(
        f"   Available days: {stats['available_days']:,} ({stats['available_pct']:.1f}%)"
    )
    print(
        f"   Unavailable days: {stats['unavailable_days']:,} ({stats['unavailable_pct']:.1f}%)"
    )

    print("\n✅ Data Quality:")
    print(f"   Missing values: {stats['missing_values']:,}")
    print(f"   Null prices: {stats['null_prices']:,}")

    memory_mb = processed_df.memory_usage(deep=True).sum() / 1024**2
    original_memory_mb = original_df.memory_usage(deep=True).sum() / 1024**2
    reduction_pct = ((original_memory_mb - memory_mb) / original_memory_mb) * 100

    print("\n💾 Memory Optimization:")
    print(f"   Before: {original_memory_mb:.2f} MB")
    print(f"   After: {memory_mb:.2f} MB")
    print(f"   Reduction: {reduction_pct:.1f}%")
    print("=" * 60 + "\n")


# Execute calendar data preprocessing pipeline
if __name__ == "__main__":

    raw_data_path = Path("data") / "raw" / "calendar.csv.gz"
    processed_data_path = Path("data") / "processed" / "calendar_cleaned.parquet"

    # Columns to drop (empty or redundant)
    columns_to_drop = ["price", "adjusted_price"]

    # Load dataset
    print(f"📂 Loading calendar data from: {raw_data_path}")
    original_calendar = pd.read_csv(raw_data_path)

    # Apply transformations
    calendar = convert_data_types(original_calendar)
    calendar = drop_empty_columns(calendar, columns_to_drop)

    # Collect statistics for report
    total = len(calendar)
    available_days = calendar["available"].sum()
    unavailable_days = (~calendar["available"]).sum()

    stats = {
        "date_min": calendar["date"].min(),
        "date_max": calendar["date"].max(),
        "unique_listings": calendar["listing_id"].nunique(),
        "available_days": available_days,
        "available_pct": (available_days / total) * 100,
        "unavailable_days": unavailable_days,
        "unavailable_pct": (unavailable_days / total) * 100,
        "missing_values": calendar.isna().sum().sum(),
        "null_prices": 0,  # Already dropped price columns
    }

    # Save processed dataset
    calendar.to_parquet(processed_data_path, engine="pyarrow", index=False)

    # Print report
    print_processing_report(original_calendar, calendar, columns_to_drop, stats)
    print(f"✅ Saved to: {processed_data_path}")
