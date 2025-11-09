import os
import glob
import pandas as pd

# ---------- CONFIG ----------
RAW_DATA_DIR = "data/v0"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "features.parquet")

# Define the columns we expect *and* need from the raw CSVs
EXPECTED_RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
# Define all columns that will be in the *final* parquet file
FINAL_COLUMNS = [
    "stock_id",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rolling_avg_10",
    "volume_sum_10",
    "target",
]


def load_and_combine_raw_data():
    """Load all CSV files, validate columns, and combine them."""
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    dfs = []

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    print(f"🔹 Found {len(all_files)} CSV files in {RAW_DATA_DIR}")

    for file in all_files:
        stock_id = os.path.basename(file).split("__")[0]

        try:
            df = pd.read_csv(file, on_bad_lines="skip", engine="python")
        except Exception as e:
            print(f"⚠️  Skipping {file}: could not read ({e})")
            continue

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # --- Stricter Validation ---
        # 1. Check if all expected columns are present *first*.
        #    This is the most important check.
        missing_cols = [col for col in EXPECTED_RAW_COLS if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Skipping {file}: file is missing headers {missing_cols}. Found: {df.columns.tolist()}")
            continue

        # 2. Find the timestamp column (from a flexible list)
        possible_names = ["timestamp", "datetime", "time", "date"]
        timestamp_col = next((c for c in possible_names if c in df.columns), None)

        # This should be guaranteed by the check above, but it's a good safeguard
        if timestamp_col is None:
            print(f"⚠️  Skipping {file}: missing a recognized timestamp column. Columns: {df.columns.tolist()}")
            continue

        # Rename timestamp column for consistency
        if timestamp_col != "timestamp":
            df.rename(columns={timestamp_col: "timestamp"}, inplace=True)

        # [REMOVED] The 'Fix missing header case' logic was here.
        # It was the source of the error as it allowed bad (HTML) files.

        # [REMOVED] The 'Ensure all required columns exist' check was here.
        # It is now redundant because we do a stricter check at the beginning.

        df["stock_id"] = stock_id
        # Only keep the columns we need for processing
        dfs.append(df[["stock_id"] + EXPECTED_RAW_COLS])
        print(f"✅ Loaded {file} ({len(df)} rows)")

    if not dfs:
        raise ValueError("No valid CSV files to process.")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def process_data(df):
    """Sort, resample, create features, and target column."""
    print("🔹 Processing data...")
    
    # --- Debugging Timestamp Conversion ---
    if df.empty:
        print("⚠️  process_data received an empty DataFrame. Check load_and_combine_raw_data.")
        return df

    print(f"Original timestamp examples (first 5 unique): {df['timestamp'].unique()[:5]}")
    
    # --- Define the expected timestamp format ---
    # Based on sample: '2017-01-02 09:15:00+05:30'
    timestamp_format = "%Y-%m-%d %H:%M:%S%z"
    
    print(f"🔹 Attempting to parse timestamps with format: {timestamp_format}")
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], 
        format=timestamp_format, 
        errors="coerce"
    )
    
    # Check how many rows became NaT (Not a Time)
    nat_count = df["timestamp"].isna().sum()
    total_count = len(df)
    
    if nat_count > 0:
        print(f"⚠️  {nat_count} out of {total_count} rows had unparseable timestamps (set to NaT).")

    # Drop any rows where timestamp conversion failed
    df.dropna(subset=["timestamp"], inplace=True)

    # *** NEW CHECK ***
    # Check if all rows were dropped after cleaning timestamps
    if df.empty:
        raise ValueError(
            "All rows were dropped after timestamp conversion. "
            "This means pd.to_datetime failed to parse *any* of the timestamp strings. "
            "Please check the 'Original timestamp examples' printed above and ensure they are valid datetimes."
        )

    # Sort by stock_id and timestamp
    df.sort_values(by=["stock_id", "timestamp"], inplace=True)

    # --- Explicit Resampling Loop (Replaces the groupby.apply) ---
    # This is more robust and avoids complex indexing issues.
    df_resampled_list = []
    
    print(f"🔹 Resampling data for {df['stock_id'].nunique()} stocks...")
    
    for stock_id, group in df.groupby("stock_id"):
        # Set timestamp as index *for this group only*
        group = group.set_index("timestamp")
        
        # Resample
        group_resampled = group.resample("1T").ffill()
        
        # Re-add the stock_id (which was lost during resample)
        group_resampled["stock_id"] = stock_id
        
        df_resampled_list.append(group_resampled)

    if not df_resampled_list:
        raise ValueError("No data to process after grouping.")

    # Combine all resampled groups
    df_resampled = pd.concat(df_resampled_list)
    
    # Reset the index to turn 'timestamp' into a column
    df_resampled.reset_index(inplace=True)
    # --- End of New Resampling Logic ---
    
    # Now, calculate features on the resampled data
    df = df_resampled # Use the resampled dataframe from now on

    # Ensure stock_id is properly filled (ffill might be needed if gaps existed)
    df['stock_id'] = df['stock_id'].ffill()

    # Rolling features
    df["rolling_avg_10"] = (
        df.groupby("stock_id")["close"]
        .transform(lambda s: s.rolling(window=10, min_periods=1).mean())
    )

    df["volume_sum_10"] = (
        df.groupby("stock_id")["volume"]
        .transform(lambda s: s.rolling(window=10, min_periods=1).sum())
    )

    # Target (t+5 close price movement)
    df["close_t_plus_5"] = df.groupby("stock_id")["close"].shift(-5)
    df["target"] = (df["close_t_plus_5"] > df["close"]).astype(int)

    # Drop rows with NaN (especially from last 5 rows per stock)
    df.dropna(inplace=True)
    
    # Drop the temporary helper column
    df.drop(columns=["close_t_plus_5"], inplace=True)

    print("✅ Data processing completed successfully.")
    
    # Ensure final columns are what we expect
    final_df = df[FINAL_COLUMNS].copy()
    
    return final_df


def save_processed_data(df):
    """Save final dataframe to parquet."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"💾 Saved processed data to {OUTPUT_FILE}")
    print("Final columns in parquet file:")
    print(df.columns.tolist())


def main():
    print("🚀 Starting data cleaning pipeline...")
    df = load_and_combine_raw_data()
    processed_df = process_data(df)
    save_processed_data(processed_df)
    print("🎯 All done!")


if __name__ == "__main__":
    main()
