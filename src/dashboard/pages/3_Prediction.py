import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def main():
    st.set_page_config(page_title="Match Prediction", page_icon="🎾", layout="wide")

    st.title("🎾 Tennis Match Prediction")
    st.markdown("---")

    # Check if model exists
    if not check_model_availability():
        st.warning(
            "⚠️ No trained model found. Please train a model first in the Model Details page."
        )
        return

    # Get model info
    model_info = get_model_info()
    display_model_info(model_info)

    # Main prediction interface
    st.header("🔮 Match Prediction")

    # Player input section
    player1_data, player2_data = get_player_input()

    if player1_data and player2_data:
        # Feature completion section
        feature_data = complete_features(
            player1_data, player2_data, model_info["features"]
        )

        # Prediction section
        if st.button("🎯 Predict Match Outcome", type="primary"):
            make_prediction(feature_data, player1_data, player2_data, model_info)


def check_model_availability():
    """Check if a trained model is available"""
    return (
        hasattr(st.session_state, "model_trained")
        and st.session_state.model_trained
        and (
            hasattr(st.session_state, "trained_model")
            or hasattr(st.session_state, "model_list")
        )
    )


def get_model_info():
    """Get information about the trained model"""
    is_multiple = getattr(st.session_state, "is_multiple_models", False)

    if is_multiple:
        models = st.session_state.model_list
        results_df = st.session_state.results_df
        algorithms = getattr(
            st.session_state,
            "algorithms_used",
            [f"Model {i+1}" for i in range(len(models))],
        )
    else:
        models = [st.session_state.trained_model]
        results_df = st.session_state.results_df
        algorithms = [getattr(st.session_state, "algorithm_used", "Trained Model")]

    feature_names = getattr(st.session_state, "feature_names", [])

    return {
        "models": models,
        "results_df": results_df,
        "algorithms": algorithms,
        "features": feature_names,
        "is_multiple": is_multiple,
    }


def display_model_info(model_info):
    """Display information about the available model(s)"""
    st.subheader("🤖 Available Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of Models", len(model_info["models"]))

    with col2:
        st.metric("Features Used", len(model_info["features"]))

    with col3:
        if model_info["is_multiple"]:
            best_accuracy = model_info["results_df"]["accuracy"].max()
            st.metric("Best Accuracy", f"{best_accuracy:.3f}")
        else:
            accuracy = model_info["results_df"]["accuracy"].iloc[0]
            st.metric("Model Accuracy", f"{accuracy:.3f}")

    # Model selection for multiple models
    selected_model_idx = 0
    if model_info["is_multiple"]:
        st.subheader("📊 Model Selection")
        col1, col2 = st.columns([1, 2])

        with col1:
            selected_model_idx = st.selectbox(
                "Select model for prediction:",
                range(len(model_info["models"])),
                format_func=lambda x: f"{model_info['algorithms'][x]} (Acc: {model_info['results_df'].iloc[x]['accuracy']:.3f})",
            )

        with col2:
            # Show selected model metrics
            selected_metrics = model_info["results_df"].iloc[selected_model_idx]
            metric_cols = st.columns(4)

            with metric_cols[0]:
                st.metric("Accuracy", f"{selected_metrics['accuracy']:.3f}")
            with metric_cols[1]:
                st.metric("AUC", f"{selected_metrics['auc']:.3f}")
            with metric_cols[2]:
                st.metric("F1 Score", f"{selected_metrics['f1']:.3f}")
            with metric_cols[3]:
                st.metric("Log Loss", f"{selected_metrics['logloss']:.3f}")

    # Store selected model
    st.session_state.selected_model_idx = selected_model_idx

    # Display features used
    with st.expander("📋 Features Used in Model"):
        if model_info["features"]:
            # Display features in columns
            feature_cols = st.columns(3)
            for i, feature in enumerate(model_info["features"]):
                with feature_cols[i % 3]:
                    st.write(f"• {feature}")
        else:
            st.info("No feature information available")


def get_player_input():
    """Get player input data"""
    st.subheader("👥 Player Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Player 1**")
        p1_first_name = st.text_input(
            "First Name", key="p1_first", placeholder="e.g., Roger"
        )
        p1_last_name = st.text_input(
            "Last Name", key="p1_last", placeholder="e.g., Federer"
        )
        p1_rank = st.number_input(
            "Current Rank", min_value=1, max_value=1000, value=1, key="p1_rank"
        )

        player1_data = (
            {
                "first_name": p1_first_name,
                "last_name": p1_last_name,
                "rank": p1_rank,
                "full_name": f"{p1_first_name} {p1_last_name}".strip(),
            }
            if p1_first_name and p1_last_name
            else None
        )

    with col2:
        st.write("**Player 2**")
        p2_first_name = st.text_input(
            "First Name", key="p2_first", placeholder="e.g., Rafael"
        )
        p2_last_name = st.text_input(
            "Last Name", key="p2_last", placeholder="e.g., Nadal"
        )
        p2_rank = st.number_input(
            "Current Rank", min_value=1, max_value=1000, value=2, key="p2_rank"
        )

        player2_data = (
            {
                "first_name": p2_first_name,
                "last_name": p2_last_name,
                "rank": p2_rank,
                "full_name": f"{p2_first_name} {p2_last_name}".strip(),
            }
            if p2_first_name and p2_last_name
            else None
        )

    return player1_data, player2_data


def complete_features(player1_data, player2_data, features):
    """Complete feature data for prediction"""
    st.subheader("📊 Feature Data Completion")

    # Initialize feature data
    feature_data = {}

    # Load training data if available
    training_data = load_training_data()

    # Categorize features
    feature_categories = categorize_features(features)

    # Process each category
    for category, category_features in feature_categories.items():
        if not category_features:
            continue

        st.write(f"**{category}**")

        if category == "Rank Features":
            # Handle rank-based features
            handle_rank_features(
                category_features, player1_data, player2_data, feature_data
            )

        elif category == "Player Statistics":
            # Handle player statistics
            handle_player_statistics(
                category_features,
                player1_data,
                player2_data,
                feature_data,
                training_data,
            )

        elif category == "Match Context":
            # Handle match context features
            handle_match_context(category_features, feature_data)

        else:
            # Handle other features
            handle_other_features(category_features, feature_data)

    return feature_data


def categorize_features(features):
    """Categorize features into different types"""
    categories = {
        "Rank Features": [],
        "Player Statistics": [],
        "Match Context": [],
        "Other Features": [],
    }

    for feature in features:
        feature_lower = feature.lower()

        if any(keyword in feature_lower for keyword in ["rank", "ranking"]):
            categories["Rank Features"].append(feature)
        elif any(
            keyword in feature_lower
            for keyword in ["win", "loss", "ace", "serve", "break", "set", "game"]
        ):
            categories["Player Statistics"].append(feature)
        elif any(
            keyword in feature_lower
            for keyword in ["surface", "tournament", "round", "best_of"]
        ):
            categories["Match Context"].append(feature)
        else:
            categories["Other Features"].append(feature)

    return categories


def handle_rank_features(features, player1_data, player2_data, feature_data):
    """Handle rank-based features"""
    col1, col2 = st.columns(2)

    for feature in features:
        if "player1" in feature.lower() or "p1" in feature.lower():
            with col1:
                feature_data[feature] = st.number_input(
                    f"{feature} (Player 1)",
                    value=float(player1_data["rank"]),
                    key=f"rank_{feature}_p1",
                )
        elif "player2" in feature.lower() or "p2" in feature.lower():
            with col2:
                feature_data[feature] = st.number_input(
                    f"{feature} (Player 2)",
                    value=float(player2_data["rank"]),
                    key=f"rank_{feature}_p2",
                )
        else:
            # Generic rank feature
            feature_data[feature] = st.number_input(
                f"{feature}",
                value=float(abs(player1_data["rank"] - player2_data["rank"])),
                key=f"rank_{feature}",
            )


def handle_player_statistics(
    features, player1_data, player2_data, feature_data, training_data
):
    """Handle player statistics features"""
    col1, col2 = st.columns(2)

    for feature in features:
        # Get latest data from training data
        p1_latest = get_latest_player_data(player1_data["full_name"], training_data)
        p2_latest = get_latest_player_data(player2_data["full_name"], training_data)

        if "player1" in feature.lower() or "p1" in feature.lower():
            with col1:
                default_value = get_feature_default_value(feature, p1_latest)
                feature_data[feature] = st.number_input(
                    f"{feature} (Player 1)",
                    value=default_value,
                    key=f"stat_{feature}_p1",
                    help=(
                        f"Latest available: {default_value}"
                        if p1_latest is not None
                        else "No historical data found"
                    ),
                )
        elif "player2" in feature.lower() or "p2" in feature.lower():
            with col2:
                default_value = get_feature_default_value(feature, p2_latest)
                feature_data[feature] = st.number_input(
                    f"{feature} (Player 2)",
                    value=default_value,
                    key=f"stat_{feature}_p2",
                    help=(
                        f"Latest available: {default_value}"
                        if p2_latest is not None
                        else "No historical data found"
                    ),
                )
        else:
            # Generic statistic
            default_value = get_feature_default_value(feature, None)
            feature_data[feature] = st.number_input(
                f"{feature}", value=default_value, key=f"stat_{feature}"
            )


def handle_match_context(features, feature_data):
    """Handle match context features"""
    for feature in features:
        feature_lower = feature.lower()

        if "surface" in feature_lower:
            feature_data[feature] = st.selectbox(
                f"{feature}",
                ["Hard", "Clay", "Grass", "Carpet"],
                key=f"context_{feature}",
            )
        elif "tournament" in feature_lower:
            feature_data[feature] = st.selectbox(
                f"{feature}",
                ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250", "Other"],
                key=f"context_{feature}",
            )
        elif "round" in feature_lower:
            feature_data[feature] = st.selectbox(
                f"{feature}",
                [
                    "Final",
                    "Semi-Final",
                    "Quarter-Final",
                    "Round of 16",
                    "Round of 32",
                    "Round of 64",
                    "Round of 128",
                ],
                key=f"context_{feature}",
            )
        elif "best_of" in feature_lower:
            feature_data[feature] = st.selectbox(
                f"{feature}", [3, 5], key=f"context_{feature}"
            )
        else:
            feature_data[feature] = st.text_input(
                f"{feature}", key=f"context_{feature}"
            )


def handle_other_features(features, feature_data):
    """Handle other features"""
    for feature in features:
        feature_data[feature] = st.number_input(
            f"{feature}", value=0.0, key=f"other_{feature}"
        )


def load_training_data():
    """Load training data from session state or file"""
    # Try to get training data from session state
    if hasattr(st.session_state, "training_data"):
        return st.session_state.training_data

    # If not available, return None
    # You can implement file loading here if needed
    return None


def get_latest_player_data(player_name, training_data):
    """Get the latest data for a specific player from training data"""
    if training_data is None:
        return None

    # Filter data for the specific player
    # This assumes your training data has player names in some format
    # Adjust the filtering logic based on your data structure
    try:
        # Example filtering - adjust based on your data structure
        player_data = training_data[
            (training_data["player1_name"] == player_name)
            | (training_data["player2_name"] == player_name)
        ]

        if not player_data.empty:
            # Return the most recent entry
            return player_data.iloc[-1]
    except:
        pass

    return None


def get_feature_default_value(feature, player_data):
    """Get default value for a feature from player data"""
    if player_data is None:
        return 0.0

    try:
        # Try to get the feature value from player data
        if feature in player_data:
            return float(player_data[feature])
    except:
        pass

    # Return sensible defaults based on feature type
    feature_lower = feature.lower()

    if "percentage" in feature_lower or "pct" in feature_lower:
        return 0.5
    elif "win" in feature_lower and "ratio" in feature_lower:
        return 0.5
    elif "age" in feature_lower:
        return 25.0
    elif "height" in feature_lower:
        return 180.0
    else:
        return 0.0


def make_prediction(feature_data, player1_data, player2_data, model_info):
    """Make prediction using the trained model"""
    try:
        # Get the selected model
        selected_idx = getattr(st.session_state, "selected_model_idx", 0)
        model = model_info["models"][selected_idx]

        # Prepare feature vector
        feature_vector = []
        for feature in model_info["features"]:
            if feature in feature_data:
                feature_vector.append(feature_data[feature])
            else:
                feature_vector.append(0.0)  # Default value

        feature_vector = np.array(feature_vector).reshape(1, -1)

        # Make prediction
        prediction = model.predict(feature_vector)[0]

        # Get prediction probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_vector)[0]
            prob_player1 = probabilities[0] if prediction == 0 else probabilities[1]
            prob_player2 = probabilities[1] if prediction == 0 else probabilities[0]
        else:
            prob_player1 = 0.6 if prediction == 0 else 0.4
            prob_player2 = 0.4 if prediction == 0 else 0.6

        # Display results
        display_prediction_results(
            prediction,
            prob_player1,
            prob_player2,
            player1_data,
            player2_data,
            model_info,
        )

        # Store prediction history
        store_prediction_history(
            player1_data, player2_data, prediction, prob_player1, prob_player2
        )

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")


def display_prediction_results(
    prediction, prob_player1, prob_player2, player1_data, player2_data, model_info
):
    """Display prediction results"""
    st.markdown("---")
    st.header("🏆 Prediction Results")

    # Determine winner
    winner = player1_data if prediction == 0 else player2_data
    loser = player2_data if prediction == 0 else player1_data
    winner_prob = prob_player1 if prediction == 0 else prob_player2

    # Display main result
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.success(f"🎯 **Predicted Winner: {winner['full_name']}**")
        st.write(f"**Confidence: {winner_prob:.1%}**")

    # Detailed probability breakdown
    st.subheader("📊 Probability Breakdown")

    # Create probability chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=[player1_data["full_name"], player2_data["full_name"]],
                y=[prob_player1, prob_player2],
                marker_color=[
                    "#1f77b4" if prediction == 0 else "#ff7f0e",
                    "#ff7f0e" if prediction == 0 else "#1f77b4",
                ],
                text=[f"{prob_player1:.1%}", f"{prob_player2:.1%}"],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Win Probability",
        xaxis_title="Player",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Player comparison
    st.subheader("⚖️ Player Comparison")

    comparison_data = []
    comparison_data.append(
        {
            "Metric": "Current Rank",
            player1_data["full_name"]: player1_data["rank"],
            player2_data["full_name"]: player2_data["rank"],
        }
    )

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # Model information
    selected_idx = getattr(st.session_state, "selected_model_idx", 0)
    algorithm = model_info["algorithms"][selected_idx]
    accuracy = model_info["results_df"].iloc[selected_idx]["accuracy"]

    st.info(f"📈 Prediction made using **{algorithm}** (Accuracy: {accuracy:.3f})")


def store_prediction_history(
    player1_data, player2_data, prediction, prob_player1, prob_player2
):
    """Store prediction in history"""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    prediction_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player1": player1_data["full_name"],
        "player2": player2_data["full_name"],
        "player1_rank": player1_data["rank"],
        "player2_rank": player2_data["rank"],
        "predicted_winner": (
            player1_data["full_name"] if prediction == 0 else player2_data["full_name"]
        ),
        "winner_probability": prob_player1 if prediction == 0 else prob_player2,
        "player1_probability": prob_player1,
        "player2_probability": prob_player2,
    }

    st.session_state.prediction_history.append(prediction_record)

    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]


# Add prediction history display
def display_prediction_history():
    """Display prediction history"""
    if (
        hasattr(st.session_state, "prediction_history")
        and st.session_state.prediction_history
    ):
        st.header("📜 Recent Predictions")

        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df = history_df.sort_values("timestamp", ascending=False)

        # Format for display
        display_history = history_df[
            [
                "timestamp",
                "player1",
                "player2",
                "predicted_winner",
                "winner_probability",
            ]
        ].copy()
        display_history["winner_probability"] = display_history[
            "winner_probability"
        ].apply(lambda x: f"{x:.1%}")
        display_history.columns = [
            "Time",
            "Player 1",
            "Player 2",
            "Predicted Winner",
            "Confidence",
        ]

        st.dataframe(display_history.head(10), use_container_width=True)


if __name__ == "__main__":
    main()

    # Display prediction history at the bottom
    st.markdown("---")
    display_prediction_history()
