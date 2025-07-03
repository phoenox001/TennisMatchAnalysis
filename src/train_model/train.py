from src.features.features import create_feature_dataframe
from src.features.features import get_preprocessed_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np


def createTrainingData():
    """
    Creates training data for the model.
    Sets type to categorical for categorical features and returns the feature matrix X and target vector y.
    """
    matches, players = get_preprocessed_data()
    trainingData = create_feature_dataframe(matches, players)

    X = trainingData.drop(
        columns=[
            "player1_wins",
            "match_id",
            "player1_dob",
            "player2_dob",
            "tourney_date",
            ""
        ]
    )
    y = trainingData["player1_wins"]

    y["player1_wins"] = y["player1_wins"].astype("category")

    catCols = [
        "player1_hand",
        "player2_hand",
        "player1_ioc",
        "player2_ioc",
        "surface",
        "best_of",
        "tourney_level",
    ]
    for col in catCols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, y


def crossValidateModel(
    X, y, n_splits=5, save_models=False, model_prefix="xgbModelFold"
):
    """
    Cross-validates the model using Stratified K-Fold cross-validation.
    Args:
        X (pd.DataFrame): Features for the model.
        y (pd.Series): Target variable indicating player 1 wins.
        n_splits (int): Number of splits for cross-validation.
    Returns:
        tuple: A tuple containing lists of accuracy, AUC, and F1 scores for each fold.
        model: The trained XGBoost model.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    models = []

    fold = 1
    for train_idx, val_idx in skf.split(X, y):
        print(f"Training fold {fold}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42 + fold,  # different seed per fold
            use_label_encoder=False,
            eval_metric="logloss",
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)

        results.append({"fold": fold, "accuracy": acc, "auc": auc, "f1": f1})

        models.append(model)

        if save_models:
            filename = f"{model_prefix}_{fold}.pkl"
            joblib.dump(model, filename)
            print(f"Saved model to {filename}")

        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}\n")

    return results, models
