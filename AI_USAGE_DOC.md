## **Utilized AI Tools** 

**Example AI Tools : (ChatGPT, Gemini, Copilot, DeepSeek Chat, Cursor, Lovable, Bolt etc)**

1. **Gemini**  
2. **ChatGPT**  
3. 

## **AI Conversation History \- Shared chat Links** 

This is a comprehensive MLOps project. I'll break down the solution into a clear, step-by-step plan covering every deliverable you mentioned.

Here is the complete roadmap to solve this problem.

---

## **Phase 1: Project Setup & Data Versioning (DVC)**

First, set up your local environment, GitHub repository, and DVC to track your two data versions.

1. **Create GitHub Repository:**  
   * Go to GitHub and create a new repository (e.g., stock-predictor-mlops).  
   * Clone it to your local machine: git clone \<your-repo-url\>  
   * cd stock-predictor-mlops  
2. **Set Up Python Environment:**  
   * Create a virtual environment: python \-m venv venv  
   * Activate it: source venv/bin/activate (or .\\venv\\Scripts\\activate on Windows)  
   * Create a requirements.txt file with the following:

```

pandas
scikit-learn
dvc[gcs]
mlflow
feast
cml
pytest
lxml

```

   *   
   * Install them: pip install \-r requirements.txt  
3. **Initialize DVC and Git:**  
   * git init (if not already done by clone)  
   * dvc init  
   * This creates a .dvc directory. Git-commit this change:  
     git commit \-m "Initialize DVC"  
4. **Structure and Version Your Data:**  
   * Create your data directories:  
   * Bash

```

mkdir -p data/v0
mkdir -p data/v1
mkdir -p data/raw # This will be the DVC-tracked dir

```

   *   
   * Place the AARTIIND... and ABCAPITAL... files into data/v0/.  
   * Place the ABFRL..., ADANIENT..., and ADANIGAS... files into data/v1/.  
5. **Create Iteration 1 (v0):**  
   * Copy the v0 data into your "live" raw directory: cp data/v0/\* data/raw/  
   * Tell DVC to track this directory: dvc add data/raw  
   * Git-commit the pointer file:  
   * Bash

```

git add data/raw.dvc .gitignore
git commit -m "feat: Add v0 data"
git tag -a "v0" -m "Data version 0"

```

   *   
6. **Create Iteration 2 (v1):**  
   * Now, add the v1 data to the raw directory: cp data/v1/\* data/raw/  
   * Update DVC: dvc add data/raw  
   * Git-commit the change:  
   * Bash

```

git add data/raw.dvc
git commit -m "feat: Add v1 data"
git tag -a "v1" -m "Data version 1"

```

   *   
   * Now you can switch between versions using git checkout v0 and git checkout v1, followed by dvc checkout.  
7. **(Optional Bonus) Configure GCS Remote:**  
   * Create a GCS bucket (e.g., my-stock-dvc-store).  
   * Add it as the DVC remote: dvc remote add \-d gcs gcs://my-stock-dvc-store/stock-data  
   * Push your data: dvc push  
   * Commit the DVC config change:  
     git add .dvc/config  
     git commit \-m "Configure GCS remote"

---

## **Phase 2: Data Processing & Feature Engineering**

Create a script (e.g., src/process.py) to handle the core logic. **This is the most important part.**

1. **Load All Files:** Read all \*.csv files from the data/raw directory. Add a stock\_id column based on the filename (e.g., "AARTIIND"). Concatenate them into a single pandas DataFrame.  
2. **Handle Time:**  
   * Convert timestamp column to datetime objects: pd.to\_datetime(df\['timestamp'\])  
   * **Crucial:** Sort the data, as it's not guaranteed to be in order: df.sort\_values(by=\['stock\_id', 'timestamp'\], inplace=True)  
3. **Fill Gaps:** This is a key requirement. You must resample the data to a 1-minute frequency for each stock and forward-fill any missing minutes.  
4. Python

```

df.set_index('timestamp', inplace=True)
df = df.groupby('stock_id').apply(
    lambda group: group.resample('1T').ffill()
)
df.reset_index(inplace=True)

```

7.   
8. **Create Features (X):**  
   * Group by stock\_id to prevent data from one stock from leaking into another's calculation.  
   * df\['rolling\_avg\_10'\] \= df.groupby('stock\_id')\['close'\].transform(lambda s: s.rolling(window=10, min\_periods=1).mean())  
   * df\['volume\_sum\_10'\] \= df.groupby('stock\_id')\['volume'\].transform(lambda s: s.rolling(window=10, min\_periods=1).sum())  
9. **Create Target (y):**  
   * This is the t+5 prediction. We use shift(-5) to pull the "close" price from 5 minutes *in the future* back to the current row t.  
   * df\['close\_t\_plus\_5'\] \= df.groupby('stock\_id')\['close'\].shift(-5)  
   * df\['target'\] \= (df\['close\_t\_plus\_5'\] \> df\['close'\]).astype(int)  
10. **Clean and Save:**  
    * Drop any rows with NaN values (especially the last 5 rows of each stock, which won't have a target). df.dropna(inplace=True)  
    * Save the final, processed data. Parquet is efficient:  
      mkdir \-p data/processed  
      df.to\_parquet('data/processed/features.parquet')

---

## **Phase 3: Feature Store Integration (Feast)**

Use Feast to define, serve, and manage your new features.

1. **Initialize Feast:**  
   * Create a feature repository: feast init feature\_repo  
   * cd feature\_repo  
2. **Define Features (in feature\_definition.py):**  
   * Modify the example file. You need:  
     * An **Entity**: stock \= Entity(name="stock\_id", join\_keys=\["stock\_id"\])  
     * A **FileSource**: Point to your processed data:  
     * Python

```

from feast import FileSource
processed_data_source = FileSource(
    path="../data/processed/features.parquet",
    timestamp_field="timestamp"
)

```

     *   
     * **Feature Views**: One for features, one for the target.  
     * Python

```

from feast import FeatureView, Feature
from google.protobuf.duration_pb2 import Duration

stock_features_view = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=Duration(days=1), # How long to keep in online store
    features=[
        Feature(name="rolling_avg_10", dtype=Float32),
        Feature(name="volume_sum_10", dtype=Float32),
    ],
    source=processed_data_source,
    online=True
)

stock_target_view = FeatureView(
    name="stock_target",
    entities=[stock],
    ttl=Duration(days=1),
    features=[Feature(name="target", dtype=Int64)],
    source=processed_data_source,
    online=False # Target is not needed for online inference
)

```

     *   
3. **Apply and Materialize:**  
   * Apply your definitions: feast apply  
   * Load your feature data from the parquet file into the online store (which is SQLite by default): feast materialize-incremental 2022-01-01T00:00:00 (Use a date in the future to capture all your data).

---

## **Phase 4: Model Training & Experiment Tracking (MLflow)**

Create a src/train.py script.

1. **Set Up MLflow:**  
   * import mlflow  
   * mlflow.set\_experiment("Stock Predictor")  
   * (Optional) Run mlflow ui in your terminal to see the dashboard.  
2. **Get Training Data from Feast:**  
   * Do *not* read the parquet file directly. Use Feast to build your training dataset.  
3. Python

```

from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")

# Get all (entity, timestamp) pairs to query
entity_df = pd.read_parquet("data/processed/features.parquet")[["stock_id", "timestamp"]]

training_job = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
        "stock_target:target"
    ]
)
training_df = training_job.to_df()

```

6.   
7. **Train-Test Split:**  
   * from sklearn.model\_selection import train\_test\_split  
   * X \= training\_df\[\['rolling\_avg\_10', 'volume\_sum\_10'\]\]  
   * y \= training\_df\['target'\]  
   * X\_train, X\_test, y\_train, y\_test \= train\_test\_split(X, y, test\_size=0.2, random\_state=42)  
8. **Hyperparameter Tuning & Tracking:**  
   * Wrap your training in an mlflow.start\_run() block.  
   * Use a simple model like RandomForestClassifier.  
   * Log params, metrics, and the model.  
9. Python

```

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

with mlflow.start_run() as run:
    # --- Hyperparameter Tuning ---
    # (Example with simple params)
    n_estimators = 100
    max_depth = 10

    # Log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # --- Train Model ---
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- Evaluate & Log Metrics ---
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    # --- Log Model ---
    mlflow.sklearn.log_model(model, "model")

    # --- Save metrics for CML ---
    with open("metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

```

12. 

---

## **Phase 5: CI/CD, Testing, and CML Reporting**

Automate your pipeline using GitHub Actions.

1. **Create Feature Tests:**  
   * Create a file tests/test\_features.py.  
   * Write pytest tests as required.  
2. Python

```

import pandas as pd

def test_rolling_avg_not_null():
    df = pd.read_parquet("data/processed/features.parquet")
    assert df['rolling_avg_10'].isnull().sum() == 0

def test_volume_sum_positive():
    df = pd.read_parquet("data/processed/features.parquet")
    assert (df['volume_sum_10'] >= 0).all()

def test_target_binary():
    df = pd.read_parquet("data/processed/features.parquet")
    assert df['target'].isin([0, 1]).all()

```

5.   
6. **Create GitHub Actions Workflow:**  
   * Create .github/workflows/run\_pipeline.yml:  
7. YAML

```

name: Run ML Pipeline

on: [push]

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0 # Needed for DVC

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Pull DVC data
        run: dvc pull data/raw -r gcs # Use -r gcs if you set it up

      - name: Process data
        run: python src/process.py

      - name: Run feature tests
        run: pytest tests/

      - name: Setup Feast
        run: |
          cd feature_repo
          feast apply
          feast materialize-incremental 2022-01-01T00:00:00
          cd ..

      - name: Train model
        run: python src/train.py

      - name: Create CML report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          echo "## Model Metrics" > report.md
          cat metrics.txt >> report.md
          cml-send-comment report.md

```

10. 

---

## **Phase 6: Executing the Two Iterations**

Your CI/CD pipeline will automatically run on the *latest* commit. To run the two iterations and compare them in MLflow:

* **Iteration 1 (v0):**  
  1. git checkout v0  
  2. dvc checkout (This pulls the v0 data into data/raw)  
  3. Manually run your scripts:  
     * python src/process.py  
     * cd feature\_repo && feast apply && feast materialize-incremental ... && cd ..  
     * python src/train.py (Make sure to set mlflow.set\_experiment("Stock Predictor v0"))  
* **Iteration 2 (v1):**  
  1. git checkout v1 (or main)  
  2. dvc checkout (This pulls the v0+v1 data into data/raw)  
  3. Run the scripts again:  
     * python src/process.py  
     * cd feature\_repo && feast apply && feast materialize-incremental ... && cd ..  
     * python src/train.py (Set mlflow.set\_experiment("Stock Predictor v1"))

Now, when you run mlflow ui, you will have two experiments (or two runs in one experiment) showing the performance difference between the v0 and v1 datasets.

## **Prompts Utilized** 

Problem Statement:

You are working as a quantitative analyst in an investment firm. You are tasked with building a predictor for stock movements using technical analysis.

Specifically, using the minute-level data, you are to predict at every minute, whether that particular stock will trade up 5 minutes later.

The features you will compute and use in your prediction are given below:

Feature Name:

1\) rolling\_avg\_10 : 10-min moving average of close price(t-9 to t)

2\) volume\_sum\_10 : Total volume traded over 10 min (t-10 to t)

where t is the time instant of the prediction.

Predicted output should be 1 if that stock will close 5 minutes later at a stock price that is higher than now, and 0 if not. That is, using the past 10 minutes of data, predict now for what would happen after 5 minutes.

prediction/target column can be created for the entire data as a separate column based on actual stock price values and use it as ground truth for training and testing.

The predictor should be trained in 2 iterations. For the first iteration, use v0 data. For the second, use both v0 and v1.

Deliverables:

\-GitHub repository

\-Set up Data versioning using DVC for the 2 versions of provided data

\-(Optional) Configuring Google Cloud Storage as Remote Storage Backend for DVC (Bonus \- 1 mark)

\-Execute training and evaluation scripts producing valid predictions for 2 iterations

\-Integrate and Utilize Feature store using Feast.

\-Configure CI with at least 1 test per feature and report generation using CML on Github.

\-Integrate Hyperparameter tuning and experiment tracking using MLflow.

No separate test data is provided. Split the provided data into train and test.

Do not assume the data is sorted in time order in the file.

If there is a gap for any minute for a stock, augment it with previous minute’s data values.

Data Description:

NSE minute-level data for stocks.

Data comprises opening price, highest price, closing price, traded volume for a stock at every minute in the period from 2018 to 2021\.

Fields : timestamp,open,high,low,close,volume

for the first iteration of the model (v0), use data in v0 data folder comprising of these stocks:

AARTIIND

ABCAPITAL

For the second iteration of the model (v1), add v1 data to v0 \- the data from v1 data folder comprising of these stocks:

ABFRL

ADANIENT

ADANIGAS

v0 has these two files

AARTIIND\_\_EQ\_\_NSE\_\_NSE\_\_MINUTE.csv

ABCAPITAL\_\_EQ\_\_NSE\_\_NSE\_\_MINUTE.csv

v1 has these three files

ABFRL\_\_EQ\_\_NSE\_\_NSE\_\_MINUTE.csv

ADANIENT\_\_EQ\_\_NSE\_\_NSE\_\_MINUTE.csv

ADANIGAS\_\_EQ\_\_NSE\_\_NSE\_\_MINUTE.csv HELP ME SOLVE THIS PROBLEM GIVE ME THE STEPS REQUIRED

