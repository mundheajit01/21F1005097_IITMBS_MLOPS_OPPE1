import mlflow
import mlflow.sklearn
from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import sys

# ==============================
# 1️⃣ Set up MLflow Experiment
# ==============================
mlflow.set_experiment("Stock Predictor")

# ==============================
# 2️⃣ Load Training Data (THE FIX)
# ==============================
print("🔍 Loading pre-processed training data...")

# We are loading the parquet file directly.
# Your `process_raw_data.py` script already created this file
# with the features (rolling_avg_10) and the target (target)
# perfectly aligned in the same rows.
#
# We DO NOT need Feast's get_historical_features() here,
# as that would cause a massive, unnecessary join of the file
# against itself (which caused your 54.5 TiB memory error).

try:
    training_df = pd.read_parquet("data/processed/features.parquet")
except FileNotFoundError:
    print("❌ ERROR: data/processed/features.parquet not found.")
    print("Please run the data processing script (e.g., process_raw_data.py) first.")
    sys.exit(1)

# We only need the columns for training
# Keep 'stock_id' and 'timestamp' if you want to see them,
# but they are not used for training.
FINAL_FEATURES = [
    "rolling_avg_10",
    "volume_sum_10",
    "target"
]

if not all(col in training_df.columns for col in FINAL_FEATURES):
    print(f"❌ ERROR: Parquet file is missing required columns.")
    print(f"Required: {FINAL_FEATURES}")
    print(f"Found: {training_df.columns.tolist()}")
    sys.exit(1)

training_df = training_df[FINAL_FEATURES]

print("✅ Training data loaded directly from parquet.")
print(f"Training data shape: {training_df.shape}")
print(training_df.head())


# ==============================
# 3️⃣ Train-Test Split
# ==============================
X = training_df[["rolling_avg_10", "volume_sum_10"]]
y = training_df["target"]

# --- (Optional) Subsample for faster development ---
# If you have millions of rows, uncomment the next lines to train on 100k
# sample_size = 100_000
# if len(X) > sample_size:
#     print(f"⚠️  Data is large. Subsampling from {len(X)} to {sample_size} rows.")
#     # We use .sample(frac=1) to shuffle before .head() to get a random sample
#     # while preserving X and y alignment, which .sample(n=...) does not.
#     shuffled_indices = X.sample(frac=1, random_state=42).index
#     sample_indices = shuffled_indices[:sample_size]
#     X, y = X.loc[sample_indices], y.loc[sample_indices]
# ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4️⃣ Train Model + Track in MLflow
# ==============================
with mlflow.start_run() as run:
    print("🚀 Starting training...")

    # --- Hyperparameters ---
    n_estimators = 100
    max_depth = 10

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # --- Train ---
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1  # <--- THIS IS THE FIX. Use all available CPU cores
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)

    # --- Log metrics ---
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    # --- Log model ---
    mlflow.sklearn.log_model(model, "model")

    # --- Save metrics for CI/CD (CML) ---
    with open("metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

    print("✅ Training complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

# ==============================
# 5️⃣ (Optional) To view MLflow dashboard:
# ==============================
# Run this command in terminal:
# mlflow ui --port 5000
