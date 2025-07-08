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


import matplotlib.pyplot as plt

# VotingClassifier does not have feature_importances_, so we aggregate from its estimators
feature_names = X.columns

importances = []
for name, estimator in ensemble_model.named_estimators_.items():
    if hasattr(estimator, "feature_importances_"):
        importances.append(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importances.append(np.abs(estimator.coef_).flatten())
    else:
        # If the estimator does not support feature importances, skip it
        continue

if importances:
    mean_importance = np.mean(importances, axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": mean_importance}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 wichtigste Features:")
    print(importance_df.head(10))
else:
    print("None of the ensemble's estimators provide feature importances.")


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
# - fix random forest and neural network
# - create dashboard
# - connect dashboard to models
# - allow for new data entries
# - allow new predictions
