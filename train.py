import mlflow
import mlflow.sklearn
from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ==============================
# 1️⃣ Set up MLflow Experiment
# ==============================
mlflow.set_experiment("Stock Predictor")

# ==============================
# 2️⃣ Load Training Data from Feast
# ==============================
print("🔍 Loading features from Feast...")

store = FeatureStore(repo_path="feature_repo/feature_repo")

# Load (entity, timestamp) pairs
entity_df = pd.read_parquet("data/processed/features.parquet")[["stock_id", "timestamp"]]

# Get historical features
training_job = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
        "stock_target:target"
    ]
)
training_df = training_job.to_df()

print("✅ Training data loaded from Feast.")
print(f"Training data shape: {training_df.shape}")
print(training_df.head())

# ==============================
# 3️⃣ Train-Test Split
# ==============================
X = training_df[["rolling_avg_10", "volume_sum_10"]]
y = training_df["target"]

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
        random_state=42
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

