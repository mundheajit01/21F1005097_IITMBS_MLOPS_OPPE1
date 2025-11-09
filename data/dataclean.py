import os
import glob
import pandas as pd

# ---------- CONFIG ----------
RAW_DATA_DIR = "data/v0"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "features.parquet")

def load_and_combine_raw_data():
    """Load all CSV files from data/raw, add stock_id, and concatenate."""
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    dfs = []

    for file in all_files:
        stock_id = os.path.basename(file).split("__")[0]  # e.g. AARTIIND from AARTIIND__EQ__NSE.csv
        df = pd.read_csv(file)

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' column not found in {file}")

        df["stock_id"] = stock_id
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def process_data(df):
    """Handle time, resampling, and feature creation."""
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by stock_id and timestamp
    df.sort_values(by=["stock_id", "timestamp"], inplace=True)

    # Set timestamp as index
    df.set_index("timestamp", inplace=True)

    # Resample each stock to 1-minute frequency and forward-fill missing data
    df = (
        df.groupby("stock_id")
        .apply(lambda group: group.resample("1T").ffill())
        .drop(columns=["stock_id"], errors="ignore")
    )

    # After groupby-apply, stock_id becomes part of index, so reset it
    df.reset_index(inplace=True)

    # Re-add stock_id if lost
    if "stock_id" not in df.columns:
        df["stock_id"] = df["stock_id"].ffill()

    # Create rolling features
    df["rolling_avg_10"] = (
        df.groupby("stock_id")["close"]
        .transform(lambda s: s.rolling(window=10, min_periods=1).mean())
    )

    df["volume_sum_10"] = (
        df.groupby("stock_id")["volume"]
        .transform(lambda s: s.rolling(window=10, min_periods=1).sum())
    )

    # Create target: predict if close_t+5 > close_t
    df["close_t_plus_5"] = (
        df.groupby("stock_id")["close"].shift(-5)
    )
    df["target"] = (df["close_t_plus_5"] > df["close"]).astype(int)

    # Drop NaN values (especially from the last few rows per stock)
    df.dropna(inplace=True)

    return df


def save_processed_data(df):
    """Save processed data to Parquet."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Processed data saved to {OUTPUT_FILE}")


def main():
    print("🔹 Loading raw data...")
    df = load_and_combine_raw_data()

    print("🔹 Processing data (sorting, resampling, features)...")
    processed_df = process_data(df)

    print("🔹 Saving processed data...")
    save_processed_data(processed_df)

    print("🎯 Done! Data processing complete.")


if __name__ == "__main__":
    main()

