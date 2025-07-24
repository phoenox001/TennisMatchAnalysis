import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


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
    player1_data, player2_data, match_data = get_player_input()

    if player1_data and player2_data and match_data:
        # Feature completion section
        feature_data = complete_features(
            player1_data, player2_data, match_data, model_info["feature_names"]
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
            (hasattr(st.session_state, "results_df"))
            or (hasattr(st.session_state, "comparison_results"))
        )
    )


def get_model_info(model=None, feature_names=None):
    """
    Extract model information including name, number of features, and feature names.

    Args:
        model: Trained sklearn/xgboost/lightgbm/catboost model
        feature_names: List of feature names used during training (optional for some models)

    Returns:
        dict: Dictionary containing model_name, num_features, and feature_names
    """
    if model is None:
        model = st.session_state.model_trained
        results_df = st.session_state.results_df
        if hasattr(st.session_state, "comparison_results"):
            comparison_results = st.session_state.comparison_results

    info = {
        "model": model,
        "model_name": "",
        "model_type": "",
        "num_features": 0,
        "feature_names": [],
    }

    # Get model name and type
    model_class = model.__class__.__name__
    model_module = model.__class__.__module__

    info["model_name"] = model_class
    info["model_type"] = f"{model_module}.{model_class}"

    # Handle different model types
    if hasattr(model, "n_features_in_"):
        # Most sklearn models have this attribute after fitting
        info["num_features"] = model.n_features_in_

        # Try to get feature names
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            info["feature_names"] = list(model.feature_names_in_)
        else:
            info["feature_names"] = st.session_state.selected_features

    # Special handling for specific models
    elif isinstance(model, xgb.XGBClassifier):
        if hasattr(model, "n_features_in_"):
            info["num_features"] = model.n_features_in_
        else:
            # Fallback: try to get from booster
            try:
                booster = model.get_booster()
                feature_names_booster = booster.feature_names
                if feature_names_booster:
                    info["num_features"] = len(feature_names_booster)
                    info["feature_names"] = feature_names_booster
            except:
                pass

        # Get feature names
        if not info["feature_names"]:
            if (
                hasattr(model, "feature_names_in_")
                and model.feature_names_in_ is not None
            ):
                info["feature_names"] = list(model.feature_names_in_)
            elif feature_names is not None:
                info["feature_names"] = (
                    feature_names[: info["num_features"]]
                    if info["num_features"]
                    else feature_names
                )
            elif info["num_features"]:
                info["feature_names"] = [
                    f"feature_{i}" for i in range(info["num_features"])
                ]

    elif isinstance(model, lgb.LGBMClassifier):
        if hasattr(model, "n_features_in_"):
            info["num_features"] = model.n_features_in_
        else:
            # Fallback: try to get from booster
            try:
                booster = model.booster_
                if booster:
                    feature_names_booster = booster.feature_name()
                    info["num_features"] = len(feature_names_booster)
                    info["feature_names"] = feature_names_booster
            except:
                pass

        # Get feature names
        if not info["feature_names"]:
            if (
                hasattr(model, "feature_names_in_")
                and model.feature_names_in_ is not None
            ):
                info["feature_names"] = list(model.feature_names_in_)
            else:
                info["feature_names"] = st.session_state.selected_features

    elif isinstance(model, CatBoostClassifier):
        if hasattr(model, "n_features_in_"):
            info["num_features"] = model.n_features_in_
        else:
            # Fallback: get from feature names
            try:
                feature_names_cat = model.feature_names_
                if feature_names_cat:
                    info["num_features"] = len(feature_names_cat)
                    info["feature_names"] = feature_names_cat
            except:
                pass

        # Get feature names
        if not info["feature_names"]:
            if hasattr(model, "feature_names_") and model.feature_names_ is not None:
                info["feature_names"] = list(model.feature_names_)
            else:
                info["feature_names"] = st.session_state.selected_features

    # Handle VotingClassifier
    elif isinstance(model, VotingClassifier):
        # Get info from the first estimator
        if model.estimators_:
            first_estimator = model.estimators_[0]
            base_info = get_model_info(first_estimator, feature_names)
            info["num_features"] = base_info["num_features"]
            info["feature_names"] = base_info["feature_names"]

        # Add ensemble information
        estimator_names = [name for name in model.estimators_]
        info["ensemble_estimators"] = estimator_names
        info["voting_type"] = getattr(model, "voting", "hard")

    # Fallback for other models
    else:
        if feature_names is not None:
            info["num_features"] = len(feature_names)
            info["feature_names"] = feature_names

    return info


def display_model_info(model_info):
    """Display information about the available model(s)"""
    st.subheader("🤖 Available Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model:", model_info["model"])

    with col2:
        st.metric("Number of Features Used", model_info["num_features"])

    with col3:
        accuracy = model_info["results_df"]["accuracy"].iloc[0]
        st.metric("Model Accuracy", f"{accuracy:.3f}")
        auc = model_info["results_df"]["auc"].iloc[0]
        st.metric("Auc:", f"{auc}")

    # Display features used
    with st.expander("📋 Features Used in Model"):
        if model_info["feature_names"]:
            # Display features in columns
            feature_cols = st.columns(3)
            for i, feature in enumerate(model_info["feature_names"]):
                with feature_cols[i % 3]:
                    st.write(f"• {feature}")
        else:
            st.info("No feature information available")


def get_player_input():
    """Get player input data"""
    st.subheader("👥 Player and Match Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Player 1**")
        p1_first_name = st.text_input(
            "First Name", key="p1_first", placeholder="e.g., Jannik"
        )
        p1_last_name = st.text_input(
            "Last Name", key="p1_last", placeholder="e.g., Sinner"
        )
        p1_rank = st.number_input(
            "Current Rank", min_value=1, max_value=5000, value=1, key="p1_rank"
        )
        p1_points = st.number_input(
            "Current ATP points",
            min_value=0,
            max_value=15000,
            value=0,
            key="p1_rank_points",
        )
        p1_is_seeded = st.toggle("Is Seeded?", False, key="p1_is_seeded")
        if p1_is_seeded:
            p1_seed = st.number_input(
                "Tournament Seed (0 = no seed)",
                min_value=1,
                max_value=32,
                value=1,
                key="p1_seed",
            )
        else:
            p1_seed = 0

        player1_data = (
            {
                "first_name": p1_first_name,
                "last_name": p1_last_name,
                "player1_rank": p1_rank,
                "player1_rank_points": p1_points,
                "player1_is_seeded": p1_is_seeded,
                "player1_seed": p1_seed,
                "full_name": f"{p1_first_name} {p1_last_name}".strip(),
            }
            if p1_first_name and p1_last_name
            else None
        )

    with col2:
        st.write("**Player 2**")
        p2_first_name = st.text_input(
            "First Name", key="p2_first", placeholder="e.g., Carlos"
        )
        p2_last_name = st.text_input(
            "Last Name", key="p2_last", placeholder="e.g., Alcaraz"
        )
        p2_rank = st.number_input(
            "Current Rank",
            min_value=1,
            max_value=1000,
            value=2,
            key="p2_rank",
        )
        p2_points = st.number_input(
            "Current ATP points",
            min_value=0,
            max_value=15000,
            value=0,
            key="p2_rank_points",
        )
        p2_is_seeded = st.toggle("Is Seeded?", False, key="p2_is_seeded")
        if p2_is_seeded:
            p2_seed = st.number_input(
                "Tournament Seed (0 = no seed)",
                min_value=1,
                max_value=32,
                value=1,
                key="p2_seed",
            )
        else:
            p2_seed = 0

        player2_data = (
            {
                "first_name": p2_first_name,
                "last_name": p2_last_name,
                "player2_rank": p2_rank,
                "player2_rank_points": p2_points,
                "player2_is_seeded": p2_is_seeded,
                "player2_seed": p2_seed,
                "full_name": f"{p2_first_name} {p2_last_name}".strip(),
            }
            if p2_first_name and p2_last_name
            else None
        )

    with col3:
        st.write("**Match Info**")
        surface = st.text_input("Surface Type", key="surface", placeholder="e.g., Clay")
        tourney_name = st.text_input(
            "Tourney", key="tourney_name", placeholder="e.g., Wimbledon"
        )
        st.write(
            "Tourney Level Overview:",
            '"G": 7,  # Grand Slam',
            '"F": 6,  # ATP Finals',
            '"M": 5,  # Masters 1000',
            '"A": 4,  # ATP 500 & 250',
            '"D": 3,  # Davis Cup',
            '"C": 2,  # Challengers',
            '"S": 1,  # Satellites/ITFs',
        )
        tourney_level = st.number_input(
            "Tourney level", min_value=1, max_value=7, value=7
        )
        round = st.selectbox(
            "Tourney Round",
            key="round",
            options=[
                "F",
                "SF",
                "QF",
                "R16",
                "R32",
                "R64",
                "R128",
                "RR",
                "BR",
                "Q1",
                "Q2",
                "Q3",
                "CR",
                "PR",
                "Q4",
                "ER",
                "",
            ],
            index=6,
        )
        best_of = st.selectbox("Best Of", key="best_of", options=[3, 5], index=0)

        match_data = (
            {
                "surface": surface,
                "tourney_name": tourney_name,
                "tourney_level": tourney_level,
                "best_of": best_of,
                "round": round,
            }
            if surface and tourney_name and round
            else None
        )

    return player1_data, player2_data, match_data


def complete_features(player1_data, player2_data, match_data, features):
    """Complete feature data for prediction"""
    st.subheader("📊 Feature Data Completion")

    # Load data if available
    training_data = load_training_data()
    player_data = load_player_data()
    atp_ranking = load_atp_ranking()

    # Initialize feature data
    feature_data = pd.DataFrame()
    feature_data["player1_id"] = get_player_id(atp_ranking, player1_data)
    feature_data["player2_id"] = get_player_id(atp_ranking, player2_data)

    # Categorize features
    feature_categories = categorize_features(features)

    # Process each category
    for category, category_features in feature_categories.items():
        if not category_features:
            continue

        st.write(f"**{category}**")
        if category == "Ranking":
            handle_rank_features(
                category_features, player1_data, player2_data, feature_data
            )

        elif category == "Player Statistics":
            # Handle player statistics
            handle_player_statistics(
                category_features,
                feature_data,
                training_data,
            )

        elif category == "Match Context":
            # Handle match context features
            handle_match_context(match_data, category_features, feature_data)

        else:
            # Handle other features
            handle_other_features(category_features, feature_data)

    return feature_data


def get_player_id(df, player_dict):
    """
    Find the player_id for a player based on their first and last name.
    This function does not check for duplicate player names. It just chooses the first occurence of the name combination.
    """
    first_name = player_dict["first_name"]
    last_name = player_dict["last_name"]

    match = df[(df["name_first"] == first_name) & (df["name_last"] == last_name)]

    if len(match) == 0:
        return None
    elif len(match) == 1:
        return match["player_id"].iloc[0]
    else:
        # Multiple matches found
        print(f"Warning: Multiple players found with name {first_name} {last_name}")
        return match["player_id"].iloc[0]


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
        if any(
            keyword in feature_lower
            for keyword in [
                "rank",
                "ranking",
                "seed",
                "seeded",
            ]
        ):
            categories["Ranking"].append(feature)
        elif any(
            keyword in feature_lower
            for keyword in [
                "Won",
                "ace",
                "svpt",
                "bp",
                "set",
                "SvGms",
                "df",
                "1stIn",
                "hand",
                "ht",
                "age",
            ]
        ):
            categories["Player Statistics"].append(feature)
        elif any(
            keyword in feature_lower
            for keyword in ["surface", "tournament", "round", "best_of", "tourney"]
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
                    f"{feature}",
                    value=float(player1_data[feature]),
                    key=f"rank_{feature}_p1",
                )
        else:
            with col2:
                feature_data[feature] = st.number_input(
                    f"{feature}",
                    value=float(player2_data[feature]),
                    key=f"rank_{feature}_p2",
                )


def handle_player_statistics(features, feature_data, historical_data):
    """Handle player statistics features"""
    col1, col2 = st.columns(2)
    player1_id = feature_data["player1_id"]
    player2_id = feature_data["player2_id"]

    # Calculate from historical data
    p1_latest = get_latest_player_data(player1_id, historical_data)
    p2_latest = get_latest_player_data(player2_id, historical_data)

    for feature in features:

        if "player1" in feature.lower() or "p1" in feature.lower():
            with col1:
                default_value = get_feature_default_value(feature, p1_latest)
                feature_data[feature] = st.number_input(
                    f"{feature} ({feature_data['player1_id']})",
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
                    f"{feature} ({feature_data['player2_id']})",
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
            default_value = get_feature_default_value(feature, historical_data)
            feature_data[feature] = st.number_input(
                f"{feature}", value=default_value, key=f"stat_{feature}"
            )


def handle_match_context(match_data, features, feature_data):
    """Handle match context features"""
    for feature in features:
        feature_lower = feature.lower()

        if "surface" in feature_lower:
            surfaces = ["Hard", "Clay", "Grass", "Carpet"]
            idx = surfaces.index(match_data["surface"])
            feature_data[feature] = st.selectbox(
                f"{feature}",
                surfaces,
                index=idx,
                key=f"context_{feature}",
            )
        elif "tourney_level" in feature_lower:
            feature_data[feature] = st.number_input(
                f"{feature}",
                min_value=1,
                max_value=7,
                value=match_data["tourney_level"],
                key=f"context_{feature}",
            )
        elif "tourney_name" in feature_lower:
            feature_data[feature_data] = st.text_input(
                f"{feature}",
                value=match_data["tourney_name"],
                key=f"context_{feature}",
            )
        elif "round" in feature_lower:
            options = [
                "F",
                "SF",
                "QF",
                "R16",
                "R32",
                "R64",
                "R128",
                "RR",
                "BR",
                "Q1",
                "Q2",
                "Q3",
                "CR",
                "PR",
                "Q4",
                "ER",
                "",
            ]
            idx = options.index(match_data["round"])
            feature_data[feature] = st.selectbox(
                f"{feature}",
                options,
                index=idx,
                key=f"context_{feature}",
            )
        elif "best_of" in feature_lower:
            options = [3, 5]
            idx = options.index(match_data["best_of"])
            feature_data[feature] = st.selectbox(
                f"{feature}", options, index=idx, key=f"context_{feature}"
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
    return None


def load_atp_ranking():
    if hasattr(st.session_state, "atp_ranking"):
        return st.session_state.atp_ranking

    # If not available, return None
    return None


def load_player_data():
    """Load training data from session state or file"""
    # Try to get player data from session state
    if hasattr(st.session_state, "training_data"):
        return st.session_state.player_data

    # If not available, return None
    return None


def get_latest_player_data(player_id, historical_data):
    """Get the latest data for a specific player from training data"""
    if historical_data is None:
        st.warning("training data is not loaded")
        return None

    # Filter data for the specific player
    try:
        # use player_id to get latest info from training data
        historical_data = historical_data.sort_values(["tourney_date", "match_num"])
        player_data = historical_data[historical_data["player_id"] == player_id]

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

    if "avg" in feature_lower or "pct" in feature_lower:
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
        model = model_info["model"]

        # Prepare feature vector
        feature_vector = []
        for feature in model_info["features"]:
            if feature in feature_data:
                feature_vector.append(feature_data[feature])
            else:
                feature_vector.append(0.0)

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
    algorithm = model_info["model_name"]
    accuracy = model_info["results_df"]["accuracy"]

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
    if len(st.session_state.prediction_history) > 10:
        st.session_state.prediction_history = st.session_state.prediction_history[-10:]


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

