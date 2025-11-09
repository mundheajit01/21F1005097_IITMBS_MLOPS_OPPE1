import pandas as pd

# Load the file without trying to select columns
df_check = pd.read_parquet("data/processed/features.parquet")

# Print the list of actual columns
print("Available columns in features.parquet:")
print(df_check.columns.tolist())
