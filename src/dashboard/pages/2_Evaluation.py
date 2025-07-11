import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import warnings

warnings.filterwarnings("ignore")


def main():
    st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

    st.title("🎯 Model Evaluation Dashboard")
    st.markdown("---")

    # Check if model data exists in session state
    if not check_model_data():
        st.warning(
            "⚠️ No trained models found. Please train a model first in the Model Details page."
        )
        return

    # Get model data
    models, results_df, is_multiple = get_model_data()

    # Display overview
    display_overview(results_df, is_multiple)

    # Main evaluation content
    if is_multiple:
        display_multiple_models_evaluation(models, results_df)
    else:
        display_single_model_evaluation(models, results_df)


def check_model_data():
    """Check if model data exists in session state"""
    return (
        hasattr(st.session_state, "model_trained")
        and st.session_state.model_trained
        and hasattr(st.session_state, "results_df")
        and st.session_state.results is not None
    )


def get_model_data():
    """Get model data from session state"""
    models = st.session_state.trained_model
    results = st.session_state.results
    comparison_results = st.session_state.comparison_results

    return models, results, comparison_results


def display_overview(results_df, is_multiple):
    """Display overview metrics"""
    st.header("📈 Performance Overview")

    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)

    if is_multiple:
        best_model_idx = results_df["accuracy"].idxmax()
        best_accuracy = results_df.loc[best_model_idx, "accuracy"]
        best_auc = results_df.loc[best_model_idx, "auc"]
        best_f1 = results_df.loc[best_model_idx, "f1"]
        avg_logloss = results_df["logloss"].mean()

        with col1:
            st.metric("Best Accuracy", f"{best_accuracy:.3f}")
        with col2:
            st.metric("Best AUC", f"{best_auc:.3f}")
        with col3:
            st.metric("Best F1 Score", f"{best_f1:.3f}")
        with col4:
            st.metric("Avg Log Loss", f"{avg_logloss:.3f}")
    else:
        with col1:
            st.metric("Accuracy", f"{results_df['accuracy'].iloc[0]:.3f}")
        with col2:
            st.metric("AUC", f"{results_df['auc'].iloc[0]:.3f}")
        with col3:
            st.metric("F1 Score", f"{results_df['f1'].iloc[0]:.3f}")
        with col4:
            st.metric("Log Loss", f"{results_df['logloss'].iloc[0]:.3f}")


def display_multiple_models_evaluation(models, results_df):
    """Display evaluation for multiple models"""
    st.header("🔍 Multiple Models Comparison")

    # Model selection
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Model Selection")
        model_names = [f"Model {i+1}" for i in range(len(models))]
        selected_model_idx = st.selectbox(
            "Select model for detailed analysis:",
            range(len(models)),
            format_func=lambda x: model_names[x],
        )

    with col2:
        st.subheader("Models Comparison")

        # Performance comparison chart
        fig = create_comparison_chart(results_df)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed analysis for selected model
    st.markdown("---")
    st.header(f"📊 Detailed Analysis - {model_names[selected_model_idx]}")

    selected_model = models[selected_model_idx]
    selected_results = results_df.iloc[selected_model_idx : selected_model_idx + 1]

    display_detailed_analysis(selected_model, selected_results)

    # Models summary table
    st.markdown("---")
    st.header("📋 Models Summary")

    # Format results for display
    display_results = results_df.copy()
    display_results.index = model_names
    display_results = display_results.round(4)

    # Highlight best performing model
    styled_df = display_results.style.highlight_max(axis=0, color="lightgreen")
    st.dataframe(styled_df, use_container_width=True)


def display_single_model_evaluation(model, results_df):
    """Display evaluation for single model"""
    st.header("🔍 Model Performance Analysis")

    display_detailed_analysis(model, results_df)


def display_detailed_analysis(model, results_df):
    """Display detailed analysis for a single model"""

    # Performance metrics visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Performance Metrics")

        # Radar chart for metrics
        fig = create_radar_chart(results_df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Metric Breakdown")

        # Bar chart for metrics
        fig = create_metrics_bar_chart(results_df)
        st.plotly_chart(fig, use_container_width=True)

    # Advanced visualizations (if model has predict_proba method)
    if hasattr(model, "predict_proba"):
        st.markdown("---")
        st.header("🎯 Advanced Model Analysis")

        # Note: These would require actual test data
        # For demonstration, showing placeholder structure
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ROC Curve")
            st.info("ROC curve would be displayed here with actual test data")
            # fig = create_roc_curve(model, X_test, y_test)
            # st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Precision-Recall Curve")
            st.info(
                "Precision-Recall curve would be displayed here with actual test data"
            )
            # fig = create_pr_curve(model, X_test, y_test)
            # st.plotly_chart(fig, use_container_width=True)

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        st.markdown("---")
        st.header("🔍 Feature Importance")

        # Get feature names (you might need to store these in session state too)
        feature_names = getattr(
            st.session_state,
            "feature_names",
            [f"Feature_{i}" for i in range(len(model.feature_importances_))],
        )

        fig = create_feature_importance_chart(model.feature_importances_, feature_names)
        st.plotly_chart(fig, use_container_width=True)


def create_comparison_chart(results_df):
    """Create comparison chart for multiple models"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Accuracy", "AUC", "F1 Score", "Log Loss"),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    model_names = [f"Model {i+1}" for i in range(len(results_df))]

    # Accuracy
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=results_df["accuracy"],
            name="Accuracy",
            marker_color="#1f77b4",
        ),
        row=1,
        col=1,
    )

    # AUC
    fig.add_trace(
        go.Bar(x=model_names, y=results_df["auc"], name="AUC", marker_color="#ff7f0e"),
        row=1,
        col=2,
    )

    # F1 Score
    fig.add_trace(
        go.Bar(
            x=model_names, y=results_df["f1"], name="F1 Score", marker_color="#2ca02c"
        ),
        row=2,
        col=1,
    )

    # Log Loss (lower is better)
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=results_df["logloss"],
            name="Log Loss",
            marker_color="#d62728",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=600, showlegend=False, title_text="Model Performance Comparison"
    )

    return fig


def create_radar_chart(results_df):
    """Create radar chart for model metrics"""
    metrics = ["accuracy", "auc", "f1"]
    values = [results_df[metric].iloc[0] for metric in metrics]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values, theta=metrics, fill="toself", name="Model Performance"
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=400,
    )

    return fig


def create_metrics_bar_chart(results_df):
    """Create bar chart for metrics"""
    metrics = ["accuracy", "auc", "f1"]
    values = [results_df[metric].iloc[0] for metric in metrics]

    fig = go.Figure(
        data=[
            go.Bar(x=metrics, y=values, marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ]
    )

    fig.update_layout(
        title="Performance Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400,
    )

    return fig


def create_feature_importance_chart(importances, feature_names):
    """Create feature importance chart"""
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    top_features = min(20, len(feature_names))  # Show top 20 features

    fig = go.Figure(
        data=[
            go.Bar(
                x=[feature_names[i] for i in indices[:top_features]],
                y=[importances[i] for i in indices[:top_features]],
                marker_color="#2ca02c",
            )
        ]
    )

    fig.update_layout(
        title="Feature Importance (Top 20)",
        xaxis_title="Features",
        yaxis_title="Importance",
        height=500,
        xaxis_tickangle=-45,
    )

    return fig


def create_roc_curve(model, X_test, y_test):
    """Create ROC curve (requires test data)"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
    )

    return fig


def create_pr_curve(model, X_test, y_test):
    """Create Precision-Recall curve (requires test data)"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400,
    )

    return fig


if __name__ == "__main__":
    main()
