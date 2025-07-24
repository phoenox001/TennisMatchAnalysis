import json
import sys
from pathlib import Path
import os
import threading
import time
import streamlit as st
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from train_model.train import (
    create_training_data,
    create_ensemble,
    cross_validate_model,
    get_best_model_from_cv,
)


st.set_page_config(page_title="Models", page_icon="🤖")


def load_training_status():
    # load training status from json file
    status_file = Path(__file__).parent / "training_status.json"

    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except:
            return None
    return None


def load_training_data():
    """Load training data from session state or file"""
    # Try to get training data from session state
    if hasattr(st.session_state, "training_data"):
        return st.session_state.training_data

    # If not available, return None
    return None


def run_training(X, y):
    # run the training algorithm in a seperate thread
    if training_data is not None:
        X, y = create_training_data(
            training_data=training_data, feature_cols=selected_features
        )
        model = False
        if selected_model != "Ensemble Model (combined model)":
            results, models = cross_validate_model(
                X, y, model_name=selected_model, save_models=True, n_splits=n_folds
            )
            result, model = get_best_model_from_cv(results, models)
            st.session_state.model_trained = model
            st.session_state.results = result

        else:
            result, comparison_results, model = create_ensemble(
                X,
                y,
                top_models=top_model_count,
                n_splits=n_folds,
                save_model=save_model,
            )
            st.session_state.model_trained = model
            st.session_state.results = result
            st.session_state.comparison_results = comparison_results
    else:
        st.warning("Error: no training data loaded")
        return


training_data = load_training_data()
status = load_training_status()

st.title("Wimbledon 2025 Predictions")
st.divider()
st.header("Models")

st.text(
    "This page let's you choose the type of model or models you want to train and "
    "gives you the opportunity to adjust the features that are used to train the model.\n"
    "There are 5 different types to choose from:\n - Random Forest\n - XGBoost\n - LightGBM\n - CatBoost\n"
    " - MLP Neural Network\n or train a combined ensemble model consisting of the best of all the model types",
)

models = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Random Forest",
    "Neural Network",
    "Ensemble Model (combined model)",
]

if training_data is not None:
    loaded = False
else:
    loaded = True


if st.checkbox("Data Loaded", loaded):

    selected_model = st.selectbox("Choose your model", models)
    selected_features = st.multiselect(
        "Choose your features", st.session_state.training_data.columns
    )
    st.session_state.selected_features = selected_features

    col1, col2, col3 = st.columns(3)
    with col1:
        save_model = st.checkbox("Save the model?")
    with col2:
        n_folds = st.slider("How many folds?", min_value=1, max_value=20, value=10)
    if selected_model == "Ensemble Model (combined model)":
        with col3:
            top_model_count = st.slider(
                "How many models to combine?", min_value=1, max_value=5, value=3
            )

    X = pd.DataFrame()
    y = pd.Series()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start training"):
            if selected_features != []:
                status_file = Path(__file__).parent / "training_status.json"
                if status_file.exists():
                    status_file.unlink()

                threading.Thread(target=run_training, args=(X, y)).start()
    with col2:
        if st.button("Refresh Status"):
            st.rerun()

    if st.checkbox("Auto-refresh", value=True):
        time.sleep(5)
        st.rerun()

if status:
    st.subheader("Training Progress")

    # Progress bar
    progress = status.get("progress", 0)
    st.progress(progress / 100)

    # Status information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Status", status.get("status", "Unknown"))

    with col2:
        st.metric(
            "current_fold",
            f"{status.get('current_fold', 0)}/{status.get('total_folds', 0)}",
        )

    with col3:
        st.metric("Progress", f"{progress:.1f}%")

    # Training metrics
    if status.get("loss") is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Loss", f"{status.get('loss', 0):.4f}")
        with col2:
            st.metric("AUC", f"{status.get('auc', 0):.4f}")

    # Display raw status for debugging
    with st.expander("Raw Status Data"):
        st.json(status)
else:
    st.info("No training in progress. Click 'Start Training' to begin.")
