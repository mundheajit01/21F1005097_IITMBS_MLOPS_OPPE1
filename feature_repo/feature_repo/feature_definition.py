from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource
from google.protobuf.duration_pb2 import Duration
from feast.types import Float32, Int64

# ==============================
# 1️⃣ Define the Entity
# ==============================
stock = Entity(
    name="stock_id",  # Primary entity for features
    join_keys=["stock_id"],
    description="Unique identifier for each stock"
)

# ==============================
# 2️⃣ Define the File Source
# ==============================
processed_data_source = FileSource(
    path="../../data/processed/features.parquet",  # Relative to this repo
    timestamp_field="timestamp",                # Time column in your dataset
)

# ==============================
# 3️⃣ Define the Stock Feature View
# ==============================
stock_features_view = FeatureView(
    name="stock_features",
    entities=[stock],
    # --- FIX ---
    # ttl=Duration(seconds=86400), # 1 day, specified in seconds
    ttl=timedelta(days=1), # 1 day, specified as a timedelta object
    # --- END FIX ---
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    source=processed_data_source,
    online=True,
)

# ==============================
# 4️⃣ Define the Target Feature View
# ==============================
stock_target_view = FeatureView(
    name="stock_target",
    entities=[stock],
    # --- FIX ---
    # ttl=Duration(seconds=86400), # 1 day, specified in seconds
    ttl=timedelta(days=1), # 1 day, specified as a timedelta object
    # --- END FIX ---
    schema=[
        Field(name="target", dtype=Int64),
    ],
    source=processed_data_source,
    online=False,  # Target not needed for online inference
)

# ==============================
# 5️⃣ Optional: registry export
# ==============================
# You can include these objects in __all__ to be discovered by Feast
__all__ = [
    "stock",
    "stock_features_view",
    "stock_target_view",
    "processed_data_source"
]
