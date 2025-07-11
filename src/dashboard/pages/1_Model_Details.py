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
    compare_all_models,
)
from features.features import create_feature_dataframe, get_preprocessed_data


st.set_page_config(page_title="Models", page_icon="🤖")


def initialize_session_state():
    """Initialize session state variables"""
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "results" not in st.session_state:
        st.session_state.results = False
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = False
    if "training_history" not in st.session_state:
        st.session_state.training_history = []
    if "training_data" not in st.session_state:
        st.session_state.training_data = pd.DataFrame()


@st.cache_data
def load_training_data():
    root_dir = Path(__file__).resolve().parent.parent.parent
    print(f"Root directory: {root_dir}")

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
    st.session_state.training_data = training_data


def load_training_status():
    """Load training status from JSON file"""
    status_file = Path(__file__).parent / "training_status.json"

    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except:
            return None
    return None


# Define thread function
def run_training(X, y):
    X, y = create_training_data(
        training_data=training_data, feature_cols=selected_features
    )
    model = None
    if (
        selected_model != "Ensemble Model (combined model)"
        and selected_model != "Compare Models"
    ):
        results, models = cross_validate_model(
            X, y, model_name=selected_model, save_models=True, n_splits=n_folds
        )
        result, model = get_best_model_from_cv(results, models)
        st.session_state.model_trained = model
        st.session_state.results = result

    elif selected_model == "Ensemble Model (combined model)":
        result, comparison_results, model = create_ensemble(
            X, y, top_models=top_model_count, n_splits=n_folds, save_model=save_model
        )
        st.session_state.model_trained = model
        st.session_state.results = result
        st.session_state.comparison_results = comparison_results
    else:
        result = compare_all_models(X, y, n_splits=n_folds, save_results=save_model)
        st.session_state.comparison_results = result


training_data = load_training_data()
status = load_training_status()

st.title("Wimbledon 2025 Predictions")
st.divider()
st.header("Models")

st.text(
    "This page let's you choose the type of model or models you want to train and gives you the opportunity to adjust the features that are used to train the model.\nThere are 5 different types to choose from:\n - Random Forest\n - XGBoost\n - LightGBM\n - CatBoost\n - MLP Neural Network\n or train a combined ensemble model consisting of the best of all the model types",
)

models = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Random Forest",
    "Neural Network",
    "Compare Models",
    "Ensemble Model (combined model)",
]

if training_data is not None:
    loaded = True
else:
    loaded = False


st.checkbox("Data Loaded", loaded)

selected_model = st.selectbox("Choose your model", models)
selected_features = st.multiselect(
    "Choose your features", st.session_state.training_data.columns
)

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
# TODO
# speichere models und übergebe sie an evaluation.py

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
