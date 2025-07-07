import sys
import os

from features.features import get_preprocessed_data, create_feature_dataframe
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
print(f"Root directory: {root_dir}")

matches, players = get_preprocessed_data(root_dir=str(root_dir))

features = create_feature_dataframe(matches, players)
features.to_parquet(
    os.path.join(root_dir, "data", "test_data.parquet"),
    index=False,
)
