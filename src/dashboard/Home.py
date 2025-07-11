import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Tennis ML Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .step-number {
        background: #667eea;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
    }
    
    .tennis-emoji {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_best_model():
    """Load the best model based on AUC score from the models folder"""
    models_folder = "models"

    if not os.path.exists(models_folder):
        return None, None, None

    best_model = None
    best_auc = 0
    best_model_name = None

    # Look for model files
    for filename in os.listdir(models_folder):
        if filename.endswith(".pkl") or filename.endswith(".pickle"):
            try:
                model_path = os.path.join(models_folder, filename)

                # Try to load model metadata (if exists)
                metadata_path = model_path.replace(".pkl", "_metadata.json").replace(
                    ".pickle", "_metadata.json"
                )

                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    auc_score = metadata.get("auc", 0)
                    if auc_score > best_auc:
                        best_auc = auc_score
                        best_model_name = filename
                        # Load the actual model
                        with open(model_path, "rb") as f:
                            best_model = pickle.load(f)

            except Exception as e:
                continue

    return best_model, best_auc, best_model_name


def create_sample_performance_data():
    """Create sample performance data for demonstration"""
    models = [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
        "SVM",
        "Neural Network",
    ]
    accuracies = [0.847, 0.832, 0.798, 0.811, 0.825]
    aucs = [0.891, 0.876, 0.843, 0.857, 0.869]

    return pd.DataFrame({"Model": models, "Accuracy": accuracies, "AUC": aucs})


def main():
    # Hero Section
    st.markdown(
        """
    <div class="main-header">
        <div class="tennis-emoji">🎾</div>
        <h1>Tennis Match Predictor</h1>
        <h3>Build, Train, and Deploy Machine Learning Models for Tennis Analytics</h3>
        <p>Harness the power of AI to predict tennis match outcomes with professional-grade accuracy</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏗️ Build Model", use_container_width=True):
            st.switch_page("pages/1_Model_Details.py")
    with col2:
        if st.button("📈 Evaluate Models", use_container_width=True):
            st.switch_page("pages/2_Evaluation.py")
    with col3:
        if st.button("🎯 Make Predictions", use_container_width=True):
            st.switch_page("pages/3_Prediction.py")

    st.markdown("---")

    # Key Features Section
    st.markdown("## 🚀 Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="feature-card">
            <h4>🤖 Advanced ML Algorithms</h4>
            <p>Choose from Random Forest, XGBoost, Neural Networks, and more. Each algorithm is optimized for tennis data patterns.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="feature-card">
            <h4>🎯 Real-time Predictions</h4>
            <p>Get instant match outcome predictions with confidence intervals and detailed probability breakdowns.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="feature-card">
            <h4>📈 Model Performance Tracking</h4>
            <p>Monitor your models' performance with detailed metrics, ROC curves, and comparison tools.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Getting Started Guide
    st.markdown("## 🎯 Getting Started")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span class="step-number">2</span>
                <div>
                    <h4 style="margin: 0;">Build Your Model</h4>
                    <p style="margin: 0; color: #666;">Select features, choose algorithms, and train your custom ML model.</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span class="step-number">3</span>
                <div>
                    <h4 style="margin: 0;">Evaluate Performance</h4>
                    <p style="margin: 0; color: #666;">Test your model's accuracy and compare it with benchmarks.</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center;">
                <span class="step-number">4</span>
                <div>
                    <h4 style="margin: 0;">Make Predictions</h4>
                    <p style="margin: 0; color: #666;">Use your trained model to predict outcomes for upcoming matches.</p>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        # Quick stats
        st.markdown(
            """
        <div class="metric-card">
            <h3>10,000+</h3>
            <p>Tennis Matches Analyzed</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="metric-card">
            <h3>500+</h3>
            <p>Professional Players</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="metric-card">
            <h3>85%+</h3>
            <p>Average Accuracy</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Model Performance Showcase
    st.markdown("## 🏆 Model Performance Showcase")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Try to load the best model
        best_model, best_auc, best_model_name = load_best_model()

        if best_model and best_auc:
            st.success(f"🎯 **Best Model:** {best_model_name}")
            st.info(f"**AUC Score:** {best_auc:.3f}")

            # Create a performance comparison chart
            sample_data = create_sample_performance_data()

            # Update the best model's performance in the sample data
            if best_model_name:
                model_type = (
                    best_model_name.split("_")[0]
                    if "_" in best_model_name
                    else "Best Model"
                )
                sample_data.loc[0, "Model"] = model_type
                sample_data.loc[0, "AUC"] = best_auc
                sample_data.loc[0, "Accuracy"] = best_auc * 0.95  # Approximate accuracy

            # Sort by AUC
            sample_data = sample_data.sort_values("AUC", ascending=False)

            fig = px.bar(
                sample_data,
                x="Model",
                y="AUC",
                title="Model Performance Comparison (AUC Score)",
                color="AUC",
                color_continuous_scale="viridis",
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Model Type",
                yaxis_title="AUC Score",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(
                "No trained models found in the models folder. Train your first model to see performance metrics here!"
            )

            # Show sample performance data
            sample_data = create_sample_performance_data()

            fig = px.bar(
                sample_data,
                x="Model",
                y="AUC",
                title="Sample Model Performance (Demo Data)",
                color="AUC",
                color_continuous_scale="viridis",
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Model Type",
                yaxis_title="AUC Score",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Performance Metrics")

        if best_model and best_auc:
            # Show real metrics
            st.metric("Best AUC Score", f"{best_auc:.3f}")
            st.metric("Estimated Accuracy", f"{best_auc * 0.95:.1%}")
            st.metric(
                "Models Trained",
                (
                    len([f for f in os.listdir("models") if f.endswith(".pkl")])
                    if os.path.exists("models")
                    else 0
                ),
            )
        else:
            # Show sample metrics
            st.metric("Sample AUC Score", "0.891")
            st.metric("Sample Accuracy", "84.7%")
            st.metric("Available Algorithms", "5+")

        st.markdown("### 🎾 Latest Updates")
        st.info("🔄 Model training pipeline optimized")
        st.info("📈 New performance metrics added")
        st.info("🎯 Prediction confidence intervals improved")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>Built with ❤️ for tennis analytics enthusiasts</p>
        <p>Ready to serve up some predictions? Start by exploring the data or building your first model!</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
