import utils.config
import sys
import os

import utils.dask_wrapper as pd
from features.features import get_preprocessed_data, create_feature_dataframe
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as p

root_dir = Path(__file__).resolve().parent.parent
print(f"Root directory: {root_dir}")

matches, players = get_preprocessed_data(root_dir=str(root_dir))

features = create_feature_dataframe(matches, players)
features.to_parquet(
    os.path.join(root_dir, "data", "test_data.parquet"),
    index=False,
)

print(f"Features shape: {features.shape}")
print(f"Features columns: {features.columns.tolist()}")

# percentage of missing values per column
percent_missing = features.isnull().sum() * 100 / len(features)
missing_value_df = p.DataFrame(
    {"column_name": features.columns, "percent_missing": percent_missing}
)
print(missing_value_df)
