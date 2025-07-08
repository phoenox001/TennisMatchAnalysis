import sys
import os

from train_model.train import (
    cross_validate_model,
    create_training_data,
    get_best_model_from_cv,
    compare_all_models,
    create_ensemble,
    analyze_categorical_cardinality,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as pd

X, y = create_training_data()
ensemble_model, comparison_results, ensemble_results = create_ensemble(
    X, y, top_models=5
)
print(ensemble_results)

# results, model = cross_validate_model(X, y, model_name="Neural Network")
# print(results)
# print(model)



# Beispiel für die Verwendung:
"""
# Daten laden
X, y = create_training_data()

# Einzelnes Modell trainieren
results_df, models = cross_validate_model(X, y, model_name='LightGBM', n_splits=10)

# Alle Modelle vergleichen
comparison_results = compare_all_models(X, y, n_splits=5)

# Ensemble erstellen
ensemble_model, comparison_results, ensemble_results = create_ensemble(X, y, top_models=3)

# Bestes Einzelmodell auswählen
best_model, best_results = get_best_model_from_cv(results_df, models, metric='auc')
"""


# TODO
# - create dashboard
# - connect dashboard to models
# - allow for new data entries
# - allow new predictions
