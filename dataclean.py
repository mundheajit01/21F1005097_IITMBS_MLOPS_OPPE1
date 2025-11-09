import os
import glob
import pandas as pd

# ---------- CONFIG ----------
RAW_DATA_DIR = "data/v0"  # path to your raw CSVs
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "features.parquet")

EXPECTED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


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

        # Detect timestamp column variants
        possible_names = ["timestamp", "datetime", "time", "date"]
        timestamp_col = next((c for c in possible_names if c in df.columns), None)

        # Fix missing header case (single unnamed column or mismatch)
        if timestamp_col is None and len(df.columns) == 1:
            print(f"⚙️  Fixing missing header in {file}")
            df = pd.read_csv(
                file,
                names=EXPECTED_COLUMNS,
                header=None,
                on_bad_lines="skip",
                engine="python",
            )
            df.columns = df.columns.str.lower()
            timestamp_col = "timestamp"

        if timestamp_col is None:
            print(f"⚠️  Skipping {file}: missing timestamp column. Columns: {df.columns.tolist()}")
            continue

        # Rename timestamp column for consistency
        df.rename(columns={timestamp_col: "timestamp"}, inplace=True)

        # Ensure all required columns exist
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Skipping {file}: missing columns {missing_cols}")
            continue

        df["stock_id"] = stock_id
        dfs.append(df)
        print(f"✅ Loaded {file} ({len(df)} rows)")

    if not dfs:
        raise ValueError("No valid CSV files to process.")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def process_data(df):
    """Sort, resample, create features, and target column."""
    print("🔹 Processing data...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    # Sort by stock_id and timestamp
    df.sort_values(by=["stock_id", "timestamp"], inplace=True)

    # Set timestamp as index
    df.set_index("timestamp", inplace=True)

    # Resample each stock to 1-minute frequency with forward-fill
    df = (
        df.groupby("stock_id")
        .apply(lambda group: group.resample("1T").ffill())
        .drop(columns=["stock_id"], errors="ignore")
    )

    # Reset index (after groupby-apply)
    df.reset_index(inplace=True)

    # Ensure stock_id column is filled (some may be lost during resampling)
    if "stock_id" not in df.columns:
        df["stock_id"] = df["stock_id"].ffill()

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

    print("✅ Data processing completed successfully.")
    return df


def save_processed_data(df):
    """Save final dataframe to parquet."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"💾 Saved processed data to {OUTPUT_FILE}")


def main():
    print("🚀 Starting data cleaning pipeline...")
    df = load_and_combine_raw_data()
    processed_df = process_data(df)
    save_processed_data(processed_df)
    print("🎯 All done!")


if __name__ == "__main__":
    main()

