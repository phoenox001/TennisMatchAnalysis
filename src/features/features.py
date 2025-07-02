# File: src/features/features.py

import pandas as pd
import numpy as np
import atp_matches
import player_rankings

"""
This module contains functions to preprocess and combine features from
ATP matches and player rankings data.
It provides functionality to create meaningful features for analysis,
such as calculating match statistics, player performance metrics,
and combining match data with player rankings.
Finally, a comprehensive dataframe is created that includes all relevant features
for each match, ready for further analysis or modeling.
"""


def get_preprocessed_data():
    """
    Reads and preprocesses ATP matches and player rankings data into a single DataFrame.

    Returns:
        pd.DataFrame: DataFrame with match statistics.
        pd.DataFrame: DataFrame with player rankings and information.
    """

    atp_matches_data = atp_matches.read_all_atp_matches("../data/atp_matches")
    player_rankings_data = player_rankings.read_all_atp_rankings("../data/atp_matches")
    players_data = player_rankings.read_atp_players()

    ranked_players = player_rankings.combine_rankings_with_players(
        player_rankings_data, players_data
    )
    ranked_players = player_rankings.preprocess_player_data(ranked_players)
    atp_matches_data = atp_matches.preprocess_atp_matches_data(atp_matches_data)
    ranked_players = player_rankings.calculate_missing_dob(
        ranked_players, atp_matches_data
    )

    return atp_matches_data, ranked_players


def add_player_info(matches, players, role):
    """
    Matches player info (rank, points, etc.) to either 'winner' or 'loser' in the match data.

    Args:
        matches (DataFrame): match data
        players (DataFrame): player ranking data
        role (str): either 'winner' or 'loser'
    Returns:
        DataFrame with enriched match data
    """
    player_id_col = f"{role}_id"
    ranking_date_col = "ranking_date"
    match_date_col = "tourney_date"

    # Rename player columns to prefixed versions
    player_cols = {
        "player_id": player_id_col,
        "rank": f"{role}_rank_official",
        "points": f"{role}_points_official",
        "name_first": f"{role}_name_first",
        "name_last": f"{role}_name_last",
        "hand": f"{role}_hand_official",
        "dob": f"{role}_dob",
        "ioc": f"{role}_ioc",
        "height": f"{role}_height_official",
        "ranking_date": f"{role}_ranking_date",
    }
    players_renamed = players.copy()
    players_renamed = players_renamed.rename(columns=player_cols)

    # Sort for merge_asof
    players_renamed = players_renamed.sort_values(
        by=[player_id_col, f"{role}_ranking_date"]
    )
    matches = matches.sort_values(by=[player_id_col, match_date_col])

    # merge_asof to match latest ranking info before match
    enriched = pd.merge_asof(
        matches,
        players_renamed,
        by=player_id_col,
        left_on=match_date_col,
        right_on=f"{role}_ranking_date",
        direction="backward",
        allow_exact_matches=True,
    )
    return enriched


def merge_all_player_info(matches, players):
    """
    Merges player information for both winner and loser roles in the matches DataFrame.

    Args:
        matches (DataFrame): DataFrame containing match data.
        players (DataFrame): DataFrame containing player information.
    Returns:
        DataFrame: Matches DataFrame enriched with player information for both winner and loser roles.
    """

    matches_with_winner = add_player_info(matches, players, role="winner")
    full_information = add_player_info(matches_with_winner, players, role="loser")

    full_information = full_information.drop(
        [
            "winner_rank_official",
            "loser_rank_official",
            "winner_points_official",
            "loser_points_official",
        ],
        errors="ignore",
    )

    return full_information


def compute_historical_averages(matches, player_col, stat_prefix):
    """
    Computes historical averages for player statistics before each match date.

    Args:
        matches (DataFrame): DataFrame containing match data.
        player_col (str): Column name for player IDs (e.g., 'winner_id', 'loser_id').
        stat_prefix (str): Prefix for the statistics columns (e.g., 'w_', 'l_').
    Returns:
        DataFrame: Matches DataFrame with added average statistics for each player.
    """

    result = matches.copy()

    # Filter relevant stat columns for this player role
    cols = [col for col in matches.columns if col.startswith(stat_prefix)]

    for col in cols:
        avg_col = f"{col}_avg"
        result[avg_col] = (
            matches.groupby(player_col)[col]
            .apply(lambda x: x.shift().expanding().mean())
            .reset_index(level=0, drop=True)
        )
    return result


def create_average_match_statistics(matches):
    """
    Creates average match statistics for each player.
    The averages are calculated based on past performances before the match date.
    Args:
        matches (DataFrame): DataFrame containing match data.
    Returns:
        DataFrame: Matches DataFrame with added average statistics for each player.
    """
    matches = matches.sort_values(by="tourney_date")

    # Calculate averages to winner and loser
    matches = compute_historical_averages(matches, "winner_id", "w_")
    matches = compute_historical_averages(matches, "loser_id", "l_")

    matches = compute_historical_averages(matches, "winner_id", "winner_seed")
    matches = compute_historical_averages(matches, "loser_id", "loser_seed")

    return matches


def randomize_player_order(matches):
    """
    Randomly flips player1 and player2 columns in the DataFrame.
    This is useful for creating a balanced dataset where the order of players does not bias the model.

    Args:
        matches (DataFrame): DataFrame containing match data with player1 and player2 columns.
    Returns:
        DataFrame: DataFrame with player1 and player2 columns randomly flipped.
    """

    matches = matches.copy()
    flip_mask = np.random.rand(len(matches)) > 0.5

    for col in matches.columns:
        if col.startswith("player1_"):
            p2_col = col.replace("player1_", "player2_")
            matches.loc[flip_mask, [col, p2_col]] = matches.loc[
                flip_mask, [p2_col, col]
            ].values

    matches.loc[flip_mask, ["player1_id", "player2_id"]] = matches.loc[
        flip_mask, ["player2_id", "player1_id"]
    ].values

    matches["player1_wins"] = (matches["player1_id"] == matches["winner"]).astype(int)

    return matches.drop(columns=["winner"])


def prepare_matches_data(matches):
    """
    Rename relevant columns from 'winner' and 'loser' to a common format.
    Adds a target column for match outcome that contains the player_id of the winner.

    Args:
        matches (DataFrame): DataFrame containing match data.
    Returns:
        DataFrame: Matches DataFrame with renamed columns for winner and loser statistics.
    """

    matches = matches.copy()

    matches["winner"] = matches["winner_id"]

    # Rename winner_ to player1_ and loser_ to player2_
    rename_map = {}
    for col in matches.columns:
        if col.startswith("winner_"):
            rename_map[col] = col.replace("winner_", "player1_")
        elif col.startswith("w_"):
            rename_map[col] = col.replace("w_", "player1_")
        elif col.startswith("loser_"):
            rename_map[col] = col.replace("loser_", "player2_")
        elif col.startswith("l_"):
            rename_map[col] = col.replace("l_", "player2_")

    matches = matches.rename(columns=rename_map)
    matches = randomize_player_order(matches)

    return matches


def final_feature_dataframe(matches, players):
    """
    Combines all features into a single DataFrame for analysis. And removes some columns that are not needed for the analysis.

    Args:
        matches (DataFrame): DataFrame containing match data with player statistics.
        players (DataFrame): DataFrame containing player information and rankings.
    Returns:
        DataFrame: Final feature DataFrame ready for analysis or modeling.
    """
    
    matches = merge_all_player_info(matches, players)
    matches = create_average_match_statistics(matches)
    matches = prepare_matches_data(matches)

    # Remove unnecessary columns that are not needed for the analysis
    player1_stats_prefix = "player1_"
    player2_stats_prefix = "player2_"

    leak_cols = ["score", "minutes", "round", "match_num"]
    stats_to_remove = [
        "ace",
        "df",
        "svpt",
        "1stIn",
        "1stWon",
        "2ndWon",
        "SvGms",
        "bpSaved",
        "bpFaced",
    ]
    cols_to_drop = []

    for stat in stats_to_remove:
        cols_to_drop.append(f"{player1_stats_prefix}{stat}")
        cols_to_drop.append(f"{player2_stats_prefix}{stat}")
    cols_to_drop.extend(leak_cols)

    cols_to_drop = [col for col in cols_to_drop if col in matches.columns]
    matches_reduced = matches.drop(columns=cols_to_drop)

    return matches_reduced

