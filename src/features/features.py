# File: src/features/features.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import features.atp_matches as atp_matches
import features.player_rankings as player_rankings

#
# This module contains functions to preprocess and combine features from
# ATP matches and player rankings data.
# It provides functionality to create meaningful features for analysis,
# such as calculating match statistics, player performance metrics,
# and combining match data with player rankings.
# Finally, a comprehensive dataframe is created that includes all relevant features
# for each match, ready for further analysis or modeling.
#


def get_preprocessed_data(root_dir):
    #
    # Reads and preprocesses ATP matches and player rankings data into a single DataFrame.
    # Returns:
    #     pd.DataFrame: DataFrame with match statistics.
    #     pd.DataFrame: DataFrame with player rankings and information.
    #

    print("Reading and preprocessing ATP matches and player rankings data...")
    atp_matches_path = f"{root_dir}/data/atp_matches"
    atp_players_path = f"{root_dir}/data/atp_matches/atp_players.csv"
    atp_matches_data = atp_matches.read_all_atp_matches(atp_matches_path)
    player_rankings_data = player_rankings.read_all_atp_rankings(atp_matches_path)
    players_data = player_rankings.read_atp_players(atp_players_path)

    ranked_players = player_rankings.combine_rankings_with_players(
        player_rankings_data, players_data
    )
    ranked_players = player_rankings.preprocess_player_data(ranked_players)
    atp_matches_data = atp_matches.preprocess_atp_matches_data(atp_matches_data)
    # ranked_players = player_rankings.calculate_missing_dob(
    #     ranked_players, atp_matches_data
    # )

    print("finished reading and preprocessing ATP matches and player rankings data.")
    return atp_matches_data, ranked_players


def add_player_info(matches, players, role):
    #
    # Matches player info (rank, points, etc.) to either 'winner' or 'loser' in the match data.
    # Args:
    #     matches (dask.DataFrame or pd.DataFrame): match data
    #     players (dask.DataFrame or pd.DataFrame): player ranking data
    #     role (str): either 'winner' or 'loser'
    # Returns:
    #     dask.DataFrame or pd.DataFrame with enriched match data
    #
    player_id_col = f"{role}_id"
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
    players_renamed = players.rename(columns=player_cols)

    # Ensure id columns have the same dtype
    matches[player_id_col] = matches[player_id_col].astype(
        players_renamed[player_id_col].dtype
    )
    players_renamed[player_id_col] = players_renamed[player_id_col].astype(
        matches[player_id_col].dtype
    )

    # Sort by keys as required for merge_asof
    matches = matches.sort_values(by=[match_date_col, player_id_col]).reset_index(
        drop=True
    )
    players_renamed = players_renamed.sort_values(
        by=[f"{role}_ranking_date", player_id_col]
    ).reset_index(drop=True)
    print(
        "Before merging players_renamed:(93)",
        type(players_renamed),
        players_renamed is None,
    )
    print("Before merging matches:(97)", type(matches), matches is None)
    # Use dask_merge_asof wrapper instead of pd.merge_asof
    enriched = pd.merge_asof(
        left=matches,
        right=players_renamed,
        by=player_id_col,
        left_on=match_date_col,
        right_on=f"{role}_ranking_date",
        direction="backward",
        allow_exact_matches=True,
    )
    return enriched


def merge_all_player_info(matches, players):
    #
    # Merges player information for both winner and loser roles in the matches DataFrame.

    # Args:
    #     matches (DataFrame): DataFrame containing match data.
    #     players (DataFrame): DataFrame containing player information.
    # Returns:
    #     DataFrame: Matches DataFrame enriched with player information for both winner and loser roles.
    #

    print("Merging player information for both winner and loser roles...")
    print("Anzahl rows data:", len(matches))
    matches_with_winner = add_player_info(matches, players, role="winner")

    full_information = add_player_info(matches_with_winner, players, role="loser")

    full_information = full_information.drop(
        [
            "winner_rank_official",
            "loser_rank_official",
            "winner_points_official",
            "loser_points_official",
            "winner_hand_official",
            "loser_hand_official",
        ],
        errors="ignore",
        axis=1,
    )

    print("Finished merging player information.")
    return full_information


def create_player_match_df(matches):
    #
    # Creates a DataFrame with one row per player per match,
    # including both winner and loser information.

    # Args:
    #     matches (DataFrame): DataFrame containing match data.
    # Returns:
    #     DataFrame: Matches DataFrame with player match history.
    #

    winner_cols = [
        col
        for col in matches.columns
        if col.startswith("w_") and not col.endswith(("date", "median", "missing"))
    ]
    loser_cols = [
        col
        for col in matches.columns
        if col.startswith("l_") and not col.endswith(("date", "median", "missing"))
    ]
    # Prepare winners data

    winners = matches[
        [
            "tourney_date",
            "match_id",
            "match_num",
            "winner_id",
            "loser_id",
            "surface",
            "tourney_level",
            "tourney_name",
            "loser_rank_points",
            "loser_rank",
            *winner_cols,
        ]
    ]

    rename_dict = {}

    for col in winners.columns:
        if col.startswith("winner_"):
            rename_dict[col] = col.replace("winner_", "player_")
        if col.startswith("loser_"):
            rename_dict[col] = col.replace("loser_", "opponent_")
        if col.startswith("w_"):
            rename_dict[col] = col.replace("w_", "player_")
        if col.startswith("l_"):
            rename_dict[col] = col.replace("l_", "player_")

    rename_dict.update(
        {
            "winner_id": "player_id",
            "loser_id": "opponent_id",
            "winner_rank": "player_rank",
            "loser_rank": "opponent_rank",
        }
    )

    winners = winners.rename(columns=rename_dict)
    winners["outcome"] = 1

    # Prepare losers data
    losers = matches[
        [
            "tourney_date",
            "match_id",
            "match_num",
            "loser_id",
            "winner_id",
            "surface",
            "tourney_level",
            "tourney_name",
            "winner_rank_points",
            "winner_rank",
            *loser_cols,
        ]
    ]
    rename_dict = {}

    for col in losers.columns:
        if col.startswith("loser_"):
            rename_dict[col] = col.replace("loser_", "player_")
        if col.startswith("winner_"):
            rename_dict[col] = col.replace("winner_", "opponent_")
        if col.startswith("w_"):
            rename_dict[col] = col.replace("w_", "player_")
        if col.startswith("l_"):
            rename_dict[col] = col.replace("l_", "player_")

    rename_dict.update(
        {
            "loser_id": "player_id",
            "winner_id": "opponent_id",
            "loser_rank": "player_rank",
            "winner_rank": "opponent_rank",
        }
    )

    losers = losers.rename(columns=rename_dict)

    losers["outcome"] = 0

    # Combine and sort
    winners = winners.reset_index(drop=True)
    losers = losers.reset_index(drop=True)
    player_matches = pd.concat([winners, losers], ignore_index=True)
    player_matches = player_matches.sort_values(
        ["player_id", "tourney_date"]
    ).reset_index(drop=True)
    print(
        "After creating player matches data:(230)",
        type(player_matches),
        player_matches is None,
    )

    print(
        "Missing values in match_id?",
        len(player_matches[player_matches["match_id"].isna()]),
    )

    return player_matches


def compute_expanding_averages(matches, player_col, col_name):
    #
    # Computes rolling averages for player statistics before each match date.

    # Args:
    #     matches (DataFrame): DataFrame containing match data.
    #     player_col (str): Column name for player IDs (e.g., 'winner_id', 'loser_id').
    #     stat_prefix (str): Prefix for the statistics columns (e.g., 'w_', 'l_').
    # Returns:
    #     DataFrame: Matches DataFrame with added average statistics for each player.
    #

    print("Computing match statistics for", col_name)
    avg_col = f"{col_name}_avg"

    matches = matches.reset_index(drop=True)

    matches[avg_col] = (
        matches.sort_values([player_col, "tourney_date"])  # Ensure correct order
        .groupby(player_col)[col_name]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    matches[avg_col] = matches[avg_col].fillna(0.0)

    matches = matches.reset_index(drop=True)
    matches = matches.sort_values(by="tourney_date")

    return matches


def compute_features(matches):
    #
    # Computes various features for each player based on their match history.

    # Args:
    #     matches (DataFrame): DataFrame containing match data
    # Returns:
    #     DataFrame: Matches DataFrame with added features for each player.
    #

    print("Computing player match features...")
    player_matches = create_player_match_df(matches)

    level_map = {
        "G": 7,  # Grand Slam
        "F": 6,  # ATP Finals
        "M": 5,  # Masters 1000
        "A": 4,  # ATP 500 & 250
        "D": 3,  # Davis Cup
        "C": 2,  # Challengers
        "S": 1,  # Satellites/ITFs
    }

    # compute rolling averages for certain columns
    player_stats = [
        col
        for col in player_matches.columns
        if col.startswith("player_") and not (col.endswith("id") or ("rank" in col))
    ]

    for col in player_stats:
        print("Computing rolling averages for", col)
        player_matches = player_matches.reset_index(drop=True)
        player_matches = compute_expanding_averages(player_matches, "player_id", col)
        player_matches = player_matches.drop(columns=col)

    # average tourney level of player
    print("Calculating average tourney level of players...")
    player_matches["tourney_level_numeric"] = player_matches["tourney_level"].map(
        level_map
    )

    player_matches = player_matches.reset_index(drop=True)
    player_matches = compute_expanding_averages(
        player_matches, "player_id", "tourney_level_numeric"
    )
    player_matches = player_matches.rename(
        columns={"tourney_level_numeric_avg": "player_tourney_level_avg"}
    )
    player_matches = player_matches.drop(columns=["tourney_level_numeric"])

    # pct matches won
    print("Calculating player percentage matches won...")

    player_matches = player_matches.sort_values(["player_id", "tourney_date"])
    player_matches = player_matches.reset_index(drop=True)

    player_matches = compute_expanding_averages(player_matches, "player_id", "outcome")
    player_matches = player_matches.rename(
        columns={"outcome_avg": "player_pct_matches_won"}
    )

    # pct grand slams won
    print("Calculating player percentage grand slams won...")
    grand_slams = player_matches[player_matches["tourney_level"] == "G"]
    grand_slams = grand_slams[["player_id", "match_id", "outcome", "tourney_date"]]

    grand_slams = grand_slams.reset_index(drop=True)
    grand_slams = compute_expanding_averages(grand_slams, "player_id", "outcome")
    grand_slams = grand_slams.rename(
        columns={"outcome_avg": "player_pct_grand_slams_won"}
    )
    grand_slams = grand_slams.drop(columns=["outcome"])

    player_matches = player_matches.merge(
        grand_slams, how="left", on=["match_id", "player_id", "tourney_date"]
    )
    player_matches["player_pct_grand_slams_won"] = player_matches[
        "player_pct_grand_slams_won"
    ].ffill()
    player_matches = player_matches.reset_index(drop=True)

    # pct Wimbledon won (assuming 'Wimbledon' in tourney_name)
    print("Calculating player percentage Wimbledon won...")
    wimbledon = player_matches[player_matches["tourney_name"] == "Wimbledon"]
    wimbledon = wimbledon[["player_id", "match_id", "outcome", "tourney_date"]]

    wimbledon = wimbledon.reset_index(drop=True)
    wimbledon = compute_expanding_averages(wimbledon, "player_id", "outcome")
    wimbledon = wimbledon.rename(columns={"outcome_avg": "player_pct_wimbledon_won"})
    wimbledon = wimbledon.drop(columns=["outcome"])

    player_matches = player_matches.merge(
        wimbledon, how="left", on=["match_id", "player_id", "tourney_date"]
    )
    player_matches["player_pct_wimbledon_won"] = player_matches[
        "player_pct_wimbledon_won"
    ].ffill()
    player_matches = player_matches.reset_index(drop=True)

    player_matches = player_matches.sort_values(["player_id", "tourney_date"])

    # years on tour
    print("Calculating years on tour...")
    first_match = (
        player_matches.groupby("player_id")[["tourney_date"]]
        .min()
        .rename(columns={"tourney_date": "first_match_date"})
        .reset_index()
    )
    print("After calculating first match:(377)", type(first_match), first_match is None)

    player_matches = player_matches.merge(first_match, on="player_id", how="left")
    player_matches["player_years_on_tour"] = (
        player_matches["tourney_date"].dt.year
        - player_matches["first_match_date"].dt.year
    )
    player_matches = player_matches.drop(columns=["first_match_date"])
    player_matches = player_matches.reset_index(drop=True)

    # average rank of opponents beaten (only won matches)
    print("Calculating average rank of opponents beaten...")
    won_matches = player_matches[player_matches["outcome"] == 1]
    won_matches = won_matches[
        [
            "player_id",
            "match_id",
            "opponent_rank",
            "opponent_rank_points",
            "tourney_date",
        ]
    ]
    won_matches = won_matches.reset_index(drop=True)
    won_matches = compute_expanding_averages(won_matches, "player_id", "opponent_rank")

    won_matches = won_matches.reset_index(drop=True)
    won_matches = compute_expanding_averages(
        won_matches, "player_id", "opponent_rank_points"
    )
    won_matches = won_matches.drop(columns=["opponent_rank", "opponent_rank_points"])
    player_matches = player_matches.merge(
        won_matches, how="left", on=["match_id", "player_id", "tourney_date"]
    )
    player_matches = player_matches.reset_index(drop=True)
    player_matches = player_matches.drop(
        columns=["opponent_rank", "opponent_rank_points"]
    )
    player_matches[["opponent_rank_avg", "opponent_rank_points_avg"]] = player_matches[
        ["opponent_rank_avg", "opponent_rank_points_avg"]
    ].ffill()
    player_matches = player_matches.rename(
        columns={
            "opponent_rank_avg": "player_avg_rank_beaten",
            "opponent_rank_points_avg": "player_avg_rank_points_beaten",
        }
    )

    # percentage matches played on surface
    # We calculate a separate percentage per surface type
    print("Calculating percentage matches played on each surface...")
    player_matches = player_matches.reset_index(drop=True)

    # one hot encode surface types, then avg every type
    player_matches["surface"] = player_matches["surface"].astype("category")
    player_matches["surface"] = player_matches["surface"].cat.set_categories(
        ["Hard", "Clay", "Grass", "Carpet", "Unknown"]
    )
    surface_dummies = pd.get_dummies(player_matches["surface"], prefix="surface")
    player_matches = pd.merge(
        left=player_matches,
        right=surface_dummies,
        left_index=True,
        right_index=True,
        how="left",
    )
    player_matches = player_matches.drop(columns=["surface"])

    surfaces = [col for col in player_matches.columns if col.startswith("surface")]
    player_matches = player_matches.sort_values(["player_id", "tourney_date"])

    for surface in surfaces:
        player_matches = player_matches.reset_index(drop=True)
        player_matches = compute_expanding_averages(
            player_matches, "player_id", surface
        )
        player_matches = player_matches.drop(columns=surface)
        player_matches = player_matches.rename(
            columns={f"{surface}_avg": f"player_{surface}_avg"}
        )

    print("Finished computing player match features.")
    return player_matches


def merge_player_features(matches, player_features_df):
    #
    # Merge player features into matches DataFrame for one player side (winner or loser),
    # without chunking.

    # Args:
    #     matches (DataFrame): Main matches DataFrame
    #     player_features_df (DataFrame): Player-match features DataFrame
    #     prefix (str): Prefix for merged feature columns (e.g., 'player1', 'player2')
    #     player_id_col (str): Column in matches for player ID (e.g., 'winner_id', 'loser_id')

    # Returns:
    #     DataFrame: Matches DataFrame enriched with player features
    #
    print(f"Merging player features...")

    # prepare winner and loser dataframes
    winners = player_features_df[player_features_df["outcome"] == 1]
    losers = player_features_df[player_features_df["outcome"] == 0]

    cols_to_merge = [col for col in player_features_df.columns if col.endswith("avg")]
    cols_to_merge += [
        "player_id",
        "tourney_date",
        "match_id",
        "match_num",
        "player_years_on_tour",
        "player_pct_wimbledon_won",
        "player_pct_grand_slams_won",
        "player_pct_matches_won",
        "player_avg_rank_beaten",
        "player_avg_rank_points_beaten",
    ]

    winners = winners[cols_to_merge]

    winner_rename_dict = {}
    for col in winners.columns:
        if col.startswith("player_") and col not in [
            "player_id",
            "tourney_date",
            "match_id",
            "match_num",
        ]:
            winner_rename_dict[col] = col.replace("player_", "player1_")
        elif col not in ["player_id", "tourney_date", "match_id", "match_num"]:
            winner_rename_dict[col] = f"player1_{col}"
    winners = winners.rename(columns=winner_rename_dict)

    losers = losers[cols_to_merge]

    loser_rename_dict = {}
    for col in losers.columns:
        if col.startswith("player_") and col not in [
            "player_id",
            "tourney_date",
            "match_id",
            "match_num",
        ]:
            loser_rename_dict[col] = col.replace("player_", "player2_")
        elif col not in ["player_id", "tourney_date", "match_id", "match_num"]:
            winner_rename_dict[col] = f"player2_{col}"
    losers = losers.rename(columns=loser_rename_dict)

    # merge winners
    merged = matches.merge(
        winners,
        how="left",
        left_on=["match_id", "match_num", "winner_id", "tourney_date"],
        right_on=["match_id", "match_num", "player_id", "tourney_date"],
    )

    # Drop duplicate player_id column
    if "player_id" in merged.columns:
        merged = merged.drop(columns=["player_id"])

    # merge losers
    full_merged = merged.merge(
        losers,
        how="left",
        left_on=["match_id", "match_num", "loser_id", "tourney_date"],
        right_on=["match_id", "match_num", "player_id", "tourney_date"],
    )

    # Drop duplicate player_id column
    if "player_id" in full_merged.columns:
        full_merged = full_merged.drop(columns=["player_id"])

    print(f"Finished merging player features.")
    return full_merged


def randomize_player_order(matches):
    #
    # Randomly flips player1 and player2 columns in the DataFrame.
    # This is useful for creating a balanced dataset where the order of players does not bias the model.

    # Args:
    #     matches (DataFrame): DataFrame containing match data with player1 and player2 columns.
    # Returns:
    #     DataFrame: DataFrame with player1 and player2 columns randomly flipped.
    #
    df = matches.copy()

    # Alle player1_ und player2_ Spalten finden
    player1_cols = [col for col in df.columns if col.startswith("player1_")]
    player2_cols = [col for col in df.columns if col.startswith("player2_")]

    # Sicherstellen, dass jede player1_ eine passende player2_ Spalte hat
    common_suffixes = [
        col.replace("player1_", "")
        for col in player1_cols
        if f"player2_{col.replace('player1_', '')}" in player2_cols
    ]

    # Flip-Maske erstellen: True = vertauschen
    flip_mask = np.random.rand(len(df)) > 0.5

    for suffix in common_suffixes:
        col1 = f"player1_{suffix}"
        col2 = f"player2_{suffix}"

        temp = df.loc[flip_mask, col1].copy()
        df.loc[flip_mask, col1] = df.loc[flip_mask, col2]
        df.loc[flip_mask, col2] = temp

    # Optional: Neue Zielvariable erstellen (z.B. player1 gewinnt?)
    if "winner" in df.columns and "player1_id" in df.columns:
        df["player1_wins"] = (df["player1_id"] == df["winner"]).astype(int)
        df = df.drop(columns=["winner"])  # wenn nicht mehr gebraucht

    return df


def prepare_matches_data(matches):
    #
    # Rename relevant columns from 'winner' and 'loser' to a common format.
    # Adds a target column for match outcome that contains the player_id of the winner.

    # Args:
    #     matches (DataFrame): DataFrame containing match data.
    # Returns:
    #     DataFrame: Matches DataFrame with renamed columns for winner and loser statistics.
    #
    print("Preparing matches data by renaming columns and setting winner...")

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

    to_remove = [col for col in matches.columns if col.endswith("median")]
    matches = matches.drop(columns=to_remove)

    print("Finished preparing matches data.")
    return matches


def create_feature_dataframe(matches, players):
    #
    # Combines all features into a single DataFrame for analysis. And removes some columns that are not needed for the analysis.

    # Args:
    #     matches (DataFrame): DataFrame containing match data with player statistics.
    #     players (DataFrame): DataFrame containing player information and rankings.
    # Returns:
    #     DataFrame: Final feature DataFrame ready for analysis or modeling.
    #

    print("Creating feature DataFrame from matches and players data...")
    player_features_path = (
        "/Users/tim/Documents/Projects/TennisMatchAnalysis/data/player_features.parquet"
    )
    matches = merge_all_player_info(matches, players)

    if os.path.exists(player_features_path):
        print("Parquet file exists, reading...")
        player_features = pd.read_parquet(player_features_path)
        print(player_features.head())
    else:
        print("File does not exist. Creating from source or skipping.")
        player_features = compute_features(matches)
        player_features.to_parquet(player_features_path)

    matches = merge_player_features(matches, player_features)

    matches = prepare_matches_data(matches)

    matches = randomize_player_order(matches)

    print("Finished creating feature DataFrame.")
    return matches
