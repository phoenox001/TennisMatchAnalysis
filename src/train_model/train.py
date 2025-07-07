import os
from pathlib import Path
from features.features import create_feature_dataframe, get_preprocessed_data
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

root_dir = Path(__file__).resolve().parent.parent.parent
print(f"Root directory: {root_dir}")


def create_training_data():
    """
    Creates training data for the model.
    Sets type to categorical for categorical features and returns the feature matrix X and target vector y.
    """

    training_data_path = (
        "/Users/tim/Documents/Projects/TennisMatchAnalysis/data/training_data.parquet"
    )
    if os.path.exists(training_data_path):
        print("Training Data Parquet file exists, reading...")
        training_data = pd.read_parquet(training_data_path)
    else:
        matches, players = get_preprocessed_data(root_dir=root_dir)
        training_data = create_feature_dataframe(matches, players)

        training_data.to_parquet(
            os.path.join(root_dir, "data", "training_data.parquet"),
            index=False,
        )

    cols_to_include = [
        col
        for col in training_data.columns
        if col.endswith("avg") or col.endswith("won") or col.endswith("beaten")
    ]
    cols_to_include += [
        "tourney_name",
        "draw_size",
        "tourney_level",
        "best_of",
        "round",
        "player1_seed",
        "player1_entry",
        "player1_hand",
        "player1_ht",
        "player1_age",
        "player1_rank",
        "player1_rank_points",
        "player1_is_seeded",
        "player1_years_on_tour",
        "player2_seed",
        "player2_entry",
        "player2_hand",
        "player2_ht",
        "player2_age",
        "player2_rank",
        "player2_rank_points",
        "player2_is_seeded",
        "player2_years_on_tour",
    ]

    X = training_data[cols_to_include].copy()

    y = training_data["player1_wins"].copy().astype("category")

    catCols = [
        "surface",
        "best_of",
        "tourney_level",
        "player1_entry",
        "player1_is_seeded",
        "player2_entry",
        "player2_is_seeded",
        "tourney_name",
        "round",
        "player1_hand",
        "player2_hand",
    ]
    for col in catCols:
        if col in X.columns:
            X[col] = X[col].astype("category")
    print("Splitted data into features and target variable")
    return X, y


def prepare_data_for_model(X, model_name):
    """
    Bereitet die Daten für spezifische Modelle vor.

    Args:
        X (pd.DataFrame): Feature DataFrame
        model_name (str): Name des Modells

    Returns:
        tuple: (X_prepared, cat_features_list)
    """
    X_prepared = X.copy()

    # Definiere kategorische Features
    categorical_features = [
        "surface",
        "best_of",
        "tourney_level",
        "player1_entry",
        "player1_is_seeded",
        "player2_entry",
        "player2_is_seeded",
        "tourney_name",
        "round",
        "player1_hand",
        "player2_hand",
    ]

    # Filtere nur existierende Spalten
    existing_cat_features = [
        col for col in categorical_features if col in X_prepared.columns
    ]

    if model_name == "CatBoost":
        # Für CatBoost: Kategorische Features als object dtype
        for col in existing_cat_features:
            X_prepared[col] = X_prepared[col].astype("object")
        return X_prepared, existing_cat_features

    elif model_name in ["XGBoost", "LightGBM"]:
        # Für XGBoost und LightGBM: Kategorische Features als category dtype
        for col in existing_cat_features:
            X_prepared[col] = X_prepared[col].astype("category")
        return X_prepared, existing_cat_features

    else:
        # Für andere Modelle: One-Hot-Encoding
        X_prepared = pd.get_dummies(
            X_prepared, columns=existing_cat_features, drop_first=True
        )
        return X_prepared, []


def get_model_configurations():
    """
    Definiert alle verfügbaren Modellkonfigurationen.
    """
    models = {
        "XGBoost": {
            "model": xgb.XGBClassifier,
            "params": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "enable_categorical": True,
            },
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier,
            "params": {
                "objective": "binary",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "n_estimators": 500,
            },
        },
        "CatBoost": {
            "model": CatBoostClassifier,
            "params": {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 8,
                "l2_leaf_reg": 3,
                "random_state": 42,
                "verbose": False,
                "thread_count": -1,
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier,
            "params": {
                "n_estimators": 500,
                "max_depth": 15,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "Neural Network": {
            "model": MLPClassifier,
            "params": {
                "hidden_layer_sizes": (128, 64, 32),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.001,
                "learning_rate": "adaptive",
                "max_iter": 1000,
                "random_state": 42,
            },
        },
    }
    return models


def cross_validate_model(
    X,
    y,
    model_name="XGBoost",
    n_splits=10,
    save_models=False,
    model_prefix=None,
    optimize_hyperparams=True,
    use_early_stopping=True,
):
    """
    Cross-validates a specific model using Stratified K-Fold cross-validation.

    Args:
        X (pd.DataFrame): Features for the model.
        y (pd.Series): Target variable indicating player 1 wins.
        model_name (str): Name of the model to use ('XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'Neural Network').
        n_splits (int): Number of splits for cross-validation.
        save_models (bool): Whether to save trained models.
        model_prefix (str): Prefix for saved model files.
        optimize_hyperparams (bool): Whether to use optimized hyperparameters.
        use_early_stopping (bool): Whether to use early stopping (only for tree-based models).

    Returns:
        tuple: A tuple containing results DataFrame and list of trained models.
    """

    models_config = get_model_configurations()

    if model_name not in models_config:
        raise ValueError(
            f"Model {model_name} not supported. Available models: {list(models_config.keys())}"
        )

    model_config = models_config[model_name]
    model_class = model_config["model"]
    model_params = model_config["params"].copy()

    if model_prefix is None:
        model_prefix = f"{model_name.lower().replace(' ', '_')}ModelFold"

    # Daten für spezifisches Modell vorbereiten
    X_prepared, cat_features = prepare_data_for_model(X, model_name)

    # Klassenverteilung prüfen
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    if pos_count > 0:
        scale_pos_weight = neg_count / pos_count
        # Nur für XGBoost, LightGBM und CatBoost
        if model_name in ["XGBoost", "LightGBM"]:
            model_params["scale_pos_weight"] = scale_pos_weight
        elif model_name == "CatBoost":
            model_params["scale_pos_weight"] = scale_pos_weight
        elif model_name == "Random Forest":
            model_params["class_weight"] = "balanced"

        print(f"Klassenverteilung: {neg_count} negative, {pos_count} positive")
        print(f"scale_pos_weight: {scale_pos_weight:.3f}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    models = []
    fold = 1

    print(f"Start cross validation mit {n_splits} Folds für {model_name}")
    print(f"Verwende optimierte Hyperparameter: {optimize_hyperparams}")
    print(
        f"Early Stopping aktiviert: {use_early_stopping and model_name in ['XGBoost', 'LightGBM', 'CatBoost']}"
    )
    print(f"Kategorische Features: {cat_features}")
    print("-" * 60)

    for train_idx, val_idx in skf.split(X_prepared, y):
        print(f"Training fold {fold}/{n_splits}...")

        X_train, X_val = X_prepared.iloc[train_idx], X_prepared.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Modell mit individueller random_state für jeden Fold
        fold_params = model_params.copy()
        fold_params["random_state"] = 42 + fold

        model = None

        # Modell-spezifische Initialisierung
        if model_name == "CatBoost":
            # CatBoost mit kategorischen Features
            fold_params["cat_features"] = cat_features
            model = model_class(**fold_params)

            if use_early_stopping:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False,
                )
            else:
                model.fit(X_train, y_train)

        elif use_early_stopping and model_name in ["XGBoost", "LightGBM"]:
            if model_name == "XGBoost":
                fold_params["early_stopping_rounds"] = 20
                model = model_class(**fold_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif model_name == "LightGBM":
                model = model_class(**fold_params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(20, verbose=False)],
                )
        else:
            model = model_class(**fold_params)
            model.fit(X_train, y_train)

        # Vorhersagen
        if model is not None:
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
        else:
            print(
                f"Warning: Model is None in fold {fold}, skipping predictions for this fold."
            )
            fold += 1
            continue

        # Metriken berechnen
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        logloss = log_loss(y_val, y_proba)

        # Anzahl der verwendeten Estimators
        n_estimators_used = model_params.get(
            "n_estimators", model_params.get("iterations", "N/A")
        )
        if use_early_stopping and model_name in ["XGBoost", "LightGBM", "CatBoost"]:
            if model is not None and hasattr(model, "best_iteration"):
                n_estimators_used = model.best_iteration
            elif model is not None and hasattr(model, "best_iteration_"):
                n_estimators_used = model.best_iteration_

        results.append(
            {
                "fold": fold,
                "accuracy": acc,
                "auc": auc,
                "f1": f1,
                "logloss": logloss,
                "n_estimators_used": n_estimators_used,
            }
        )

        models.append(model)

        # Modell speichern falls gewünscht
        if save_models:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(f"{root_dir}/models", exist_ok=True)
            filename = f"{root_dir}/models/{model_prefix}_{fold}_{timestamp}.pkl"
            joblib.dump(model, filename)
            print(f"Saved model to {filename}")

        print(
            f"Fold {fold} - Accuracy: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | LogLoss: {logloss:.4f}"
        )

        if use_early_stopping and hasattr(model, "best_iteration"):
            print(f"Best iteration: {model.best_iteration}")

        print("-" * 40)
        fold += 1

    # Zusammenfassung der Ergebnisse
    results_df = pd.DataFrame(results)
    print(f"\n{model_name} Cross-Validation Ergebnisse:")
    print(
        f"Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}"
    )
    print(f"AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"F1: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    print(
        f"LogLoss: {results_df['logloss'].mean():.4f} ± {results_df['logloss'].std():.4f}"
    )

    if use_early_stopping and model_name in ["XGBoost", "LightGBM", "CatBoost"]:
        avg_estimators = results_df["n_estimators_used"].mean()
        print(f"Durchschnittliche Anzahl Bäume: {avg_estimators:.1f}")

    return results_df, models


def compare_all_models(X, y, n_splits=5, save_results=True):
    """
    Vergleicht alle verfügbaren Modelle und gibt eine Übersicht zurück.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        n_splits (int): Number of CV splits
        save_results (bool): Whether to save comparison results

    Returns:
        pd.DataFrame: Comparison results
    """

    models_config = get_model_configurations()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "roc_auc", "f1", "neg_log_loss"]

    results = {}

    print("Vergleiche alle Modelle...")
    print("=" * 80)

    for model_name, config in models_config.items():
        print(f"Trainiere {model_name}...")

        try:
            # Daten für spezifisches Modell vorbereiten
            X_prepared, cat_features = prepare_data_for_model(X, model_name)

            model_class = config["model"]
            model_params = config["params"].copy()

            # CatBoost spezielle Behandlung
            if model_name == "CatBoost":
                model_params["cat_features"] = cat_features

            # Klassengewichtung anpassen
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            if pos_count > 0:
                scale_pos_weight = neg_count / pos_count
                if model_name in ["XGBoost", "LightGBM"]:
                    model_params["scale_pos_weight"] = scale_pos_weight
                elif model_name == "CatBoost":
                    model_params["scale_pos_weight"] = scale_pos_weight
                elif model_name == "Random Forest":
                    model_params["class_weight"] = "balanced"

            model = model_class(**model_params)

            # Cross-Validation durchführen
            cv_results = cross_validate(
                model,
                X_prepared,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1,
            )

            # Ergebnisse sammeln
            results[model_name] = {
                "accuracy": cv_results["test_accuracy"].mean(),
                "accuracy_std": cv_results["test_accuracy"].std(),
                "auc": cv_results["test_roc_auc"].mean(),
                "auc_std": cv_results["test_roc_auc"].std(),
                "f1": cv_results["test_f1"].mean(),
                "f1_std": cv_results["test_f1"].std(),
                "logloss": -cv_results["test_neg_log_loss"].mean(),
                "logloss_std": cv_results["test_neg_log_loss"].std(),
            }

            print(f"✓ {model_name} abgeschlossen")

        except Exception as e:
            print(f"✗ {model_name} fehlgeschlagen: {str(e)}")
            continue

    # Ergebnisse als DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)

    # Sortiere nach AUC
    results_df = results_df.sort_values("auc", ascending=False)

    print("\nModell-Vergleich (sortiert nach AUC):")
    print("=" * 90)
    print(f"{'Modell':<15} {'Accuracy':<15} {'AUC':<15} {'F1':<15} {'LogLoss':<15}")
    print("-" * 90)

    for model_name, row in results_df.iterrows():
        print(
            f"{model_name:<15} "
            f"{row['accuracy']:.4f}±{row['accuracy_std']:.3f}   "
            f"{row['auc']:.4f}±{row['auc_std']:.3f}   "
            f"{row['f1']:.4f}±{row['f1_std']:.3f}   "
            f"{row['logloss']:.4f}±{row['logloss_std']:.3f}"
        )

    # Ergebnisse speichern
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"{root_dir}/results", exist_ok=True)
        results_df.to_csv(f"{root_dir}/results/model_comparison_{timestamp}.csv")
        print(
            f"\nErgebnisse gespeichert: {root_dir}/results/model_comparison_{timestamp}.csv"
        )

    return results_df


def create_ensemble(X, y, top_models=3, n_splits=10, save_model=True):
    """
    Erstellt ein Ensemble aus den besten Modellen.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        top_models (int): Number of top models to include in ensemble
        n_splits (int): Number of CV splits
        save_model (bool): Whether to save the ensemble model

    Returns:
        tuple: (ensemble_model, comparison_results, ensemble_results)
    """

    # Erst alle Modelle vergleichen
    print("Vergleiche Modelle für Ensemble-Auswahl...")
    comparison_results = compare_all_models(X, y, n_splits=n_splits, save_results=False)

    # Top Modelle auswählen
    top_model_names = comparison_results.head(top_models).index.tolist()
    print(f"\nErstelle Ensemble aus Top {top_models} Modellen: {top_model_names}")

    # Modelle für Ensemble erstellen
    models_config = get_model_configurations()
    ensemble_estimators = []

    # Für Ensemble verwenden wir One-Hot-Encoding für alle Modelle
    X_ensemble = X.copy()
    categorical_features = [
        "surface",
        "best_of",
        "tourney_level",
        "player1_entry",
        "player1_is_seeded",
        "player2_entry",
        "player2_is_seeded",
        "tourney_name",
        "round",
        "player1_hand",
        "player2_hand",
    ]
    existing_cat_features = [
        col for col in categorical_features if col in X_ensemble.columns
    ]
    X_ensemble = pd.get_dummies(
        X_ensemble, columns=existing_cat_features, drop_first=True
    )

    for model_name in top_model_names:
        if model_name in models_config:
            config = models_config[model_name]
            model_class = config["model"]
            model_params = config["params"].copy()

            # Für Ensemble keine kategorischen Features verwenden
            if "cat_features" in model_params:
                del model_params["cat_features"]
            if "enable_categorical" in model_params:
                model_params["enable_categorical"] = False

            # Klassengewichtung anpassen
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            if pos_count > 0:
                scale_pos_weight = neg_count / pos_count
                if model_name in ["XGBoost", "LightGBM"]:
                    model_params["scale_pos_weight"] = scale_pos_weight
                elif model_name == "CatBoost":
                    model_params["scale_pos_weight"] = scale_pos_weight
                elif model_name == "Random Forest":
                    model_params["class_weight"] = "balanced"

            model = model_class(**model_params)
            ensemble_estimators.append((model_name, model))

    # Ensemble erstellen
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting="soft")

    # Ensemble testen
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "roc_auc", "f1", "neg_log_loss"]

    ensemble_cv_results = cross_validate(
        ensemble, X_ensemble, y, cv=cv, scoring=scoring, return_train_score=False
    )

    ensemble_results = {
        "accuracy": ensemble_cv_results["test_accuracy"].mean(),
        "accuracy_std": ensemble_cv_results["test_accuracy"].std(),
        "auc": ensemble_cv_results["test_roc_auc"].mean(),
        "auc_std": ensemble_cv_results["test_roc_auc"].std(),
        "f1": ensemble_cv_results["test_f1"].mean(),
        "f1_std": ensemble_cv_results["test_f1"].std(),
        "logloss": -ensemble_cv_results["test_neg_log_loss"].mean(),
        "logloss_std": ensemble_cv_results["test_neg_log_loss"].std(),
    }

    print(f"\nEnsemble Ergebnisse:")
    print(
        f"Accuracy: {ensemble_results['accuracy']:.4f} ± {ensemble_results['accuracy_std']:.4f}"
    )
    print(f"AUC: {ensemble_results['auc']:.4f} ± {ensemble_results['auc_std']:.4f}")
    print(f"F1: {ensemble_results['f1']:.4f} ± {ensemble_results['f1_std']:.4f}")
    print(
        f"LogLoss: {ensemble_results['logloss']:.4f} ± {ensemble_results['logloss_std']:.4f}"
    )

    # Ensemble trainieren auf allen Daten
    ensemble.fit(X_ensemble, y)

    # Ensemble speichern
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"{root_dir}/models", exist_ok=True)
        filename = f"{root_dir}/models/ensemble_model_{timestamp}.pkl"
        joblib.dump(ensemble, filename)
        print(f"\nEnsemble gespeichert: {filename}")

    return ensemble, comparison_results, ensemble_results


def get_best_model_from_cv(results_df, models, metric="auc"):
    """
    Wählt das beste Modell basierend auf einer Metrik aus.

    Args:
        results_df: DataFrame mit CV-Ergebnissen
        models: Liste der trainierten Modelle
        metric: Metrik zur Auswahl ('auc', 'accuracy', 'f1')

    Returns:
        Bestes Modell und dessen Ergebnisse
    """
    best_fold_idx = results_df[metric].idxmax()
    best_model = models[best_fold_idx]
    best_results = results_df.iloc[best_fold_idx]

    print(f"Bestes Modell (Fold {best_results['fold']}) basierend auf {metric}:")
    print(f"Accuracy: {best_results['accuracy']:.4f}")
    print(f"AUC: {best_results['auc']:.4f}")
    print(f"F1: {best_results['f1']:.4f}")
    print(f"LogLoss: {best_results['logloss']:.4f}")

    return best_model, best_results
