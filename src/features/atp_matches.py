# File: src/features/atp_matches.py

import pandas as pd
import numpy as np
import os
import glob

#
# This module contains functions to read and preprocess ATP matches data.
# It provides functionality to read all ATP match files from a specified directory,
# combine them into a single DataFrame, and perform necessary preprocessing steps
# such as handling missing values, renaming columns, and adding match IDs.
# It also includes functions to impute missing match statistics based on context
# and to preprocess the matches data for further analysis.
#


def read_all_atp_matches(directory, file_pattern="atp_matches_*.csv"):
    #
    # Reads all ATP match files from a directory and concatenates them into a single DataFrame.

    # Args:
    #     directory (str): Directory containing ATP match CSV files.

    # Returns:
    #     pd.DataFrame: Concatenated DataFrame of all ATP matches.
    #

    print(f"Reading ATP matches from directory: {directory}")
    pattern = os.path.join(directory, file_pattern)
    file_list = glob.glob(pattern)

    if not file_list:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in directory '{directory}'."
        )

    atp_matches_list = [
        pd.read_csv(
            file,
            dtype={
                "l_1stIn": "float64",
                "l_1stWon": "float64",
                "l_2ndWon": "float64",
                "l_SvGms": "float64",
                "l_ace": "float64",
                "l_bpFaced": "float64",
                "l_bpSaved": "float64",
                "l_df": "float64",
                "l_svpt": "float64",
                "loser_ht": "float64",
                "loser_rank": "float64",
                "loser_rank_points": "float64",
                "minutes": "float64",
                "w_1stIn": "float64",
                "w_1stWon": "float64",
                "w_2ndWon": "float64",
                "w_SvGms": "float64",
                "w_ace": "float64",
                "w_bpFaced": "float64",
                "w_bpSaved": "float64",
                "w_df": "float64",
                "w_svpt": "float64",
                "winner_ht": "float64",
                "winner_rank": "float64",
                "winner_rank_points": "float64",
                "loser_entry": "object",
                "winner_entry": "object",
                "draw_size": "string",
                "loser_hand": "object",
                "winner_hand": "object",
                "winner_id": "int64",
                "loser_id": "int64",
                "tourney_id": "object",
                "match_num": "int64",
                "tourney_name": "object",
                "tourney_level": "object",
                "winner_ioc": "object",
                "loser_ioc": "object",
            },
        )
        for file in file_list
    ]
    combined_atp_matches = pd.concat(atp_matches_list, ignore_index=True)
    combined_atp_matches[["tourney_id", "match_num"]] = combined_atp_matches[
        ["tourney_id", "match_num"]
    ].astype(str)
    combined_atp_matches["match_id"] = (
        combined_atp_matches["tourney_id"]
        + "-"
        + combined_atp_matches["match_num"].astype("str")
    )
    print(f"Combined ATP matches from {len(file_list)} files.")
    print(f"Columns in combined matches: {combined_atp_matches.columns.tolist()}")
    return combined_atp_matches


def impute_missing_match_stats(
    matches, stat_columns, group_context=["best_of", "surface"]
):
    #
    # Imputes missing match statistics using medians grouped by match context
    # and adds _missing flags.

    # Args:
    #     matches: (pd.DataFrame) DataFrame containing match statistics.
    #     stat_columns: list of columns to impute (e.g. ['w_ace', 'l_df', ...])
    #     group_context: list of context columns (default: ['best_of', 'surface'])

    # Returns:
    #     DataFrame with imputed values and added _missing flags
    #
    print(f"Imputing missing match statistics for columns: {stat_columns}")
    matches = matches.copy()

    for col in stat_columns:

        grouped_medians = (
            matches.groupby(group_context)[col]
            .median()
            .reset_index()
            .rename(columns={col: f"{col}_median"})
        )

        matches = matches.merge(grouped_medians, on=group_context, how="left")

        matches[col] = matches[col].fillna(matches[f"{col}_median"])

        global_median = matches[col].median()
        matches[col] = matches[col].fillna(global_median)

        matches.drop(columns=[f"{col}_median"])

    print(f"Imputed missing statistics for matches.")
    return matches


def preprocess_atp_matches_data(matches):
    #
    # Cleans ATP matches data by removing duplicates, filling missing data , setting data types, and adding match_id.

    # Args:
    #     matches (pd.DataFrame): DataFrame containing ATP matches.

    # Returns:
    #     pd.DataFrame: Cleaned DataFrame with unique matches and match_id.
    #

    print("Preprocessing ATP matches data...")
    matches = matches.drop_duplicates(subset=["match_id"], keep="first")

    matches["winner_id"] = matches["winner_id"].astype("int64")
    matches["loser_id"] = matches["loser_id"].astype("int64")

    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d")
    matches["winner_is_seeded"] = pd.notna(matches["winner_seed"]).astype(int)
    matches["loser_is_seeded"] = pd.notna(matches["loser_seed"]).astype(int)
    matches["winner_seed"] = matches["winner_seed"].fillna(0).astype(int)
    matches["loser_seed"] = matches["loser_seed"].fillna(0).astype(int)

    matches["winner_entry"] = matches["winner_entry"].fillna("DE").astype(str)
    matches["loser_entry"] = matches["loser_entry"].fillna("DE").astype(str)

    matches["winner_ht"] = (
        matches["winner_ht"].fillna(matches["winner_ht"].median()).astype(float)
    )
    matches["loser_ht"] = (
        matches["loser_ht"].fillna(matches["loser_ht"].median()).astype(float)
    )

    matches["winner_age"] = (
        matches["winner_age"].fillna(matches["winner_age"].median()).astype(float)
    )
    matches["loser_age"] = (
        matches["loser_age"].fillna(matches["loser_age"].median()).astype(float)
    )

    matches["winner_hand"] = matches["winner_hand"].fillna("U").astype(str)
    matches["loser_hand"] = matches["loser_hand"].fillna("U").astype(str)

    matches["surface"] = matches["surface"].fillna("Unknown").astype(str)

    matches["draw_size"] = matches["draw_size"].replace("R", np.nan)
    matches["draw_size"] = matches["draw_size"].fillna(-1).astype(int)
    draw_size_map = matches.groupby("tourney_level")["draw_size"].median()
    draw_size_map_df = draw_size_map.reset_index().rename(
        columns={"draw_size": "draw_size_median"}
    )
    matches = matches.merge(draw_size_map_df, on="tourney_level", how="left")
    matches["draw_size"] = matches["draw_size"].mask(
        matches["draw_size"] == -1, matches["draw_size_median"]
    )
    matches = matches.drop("draw_size_median", axis=1)

    matches["winner_rank_points"] = matches["winner_rank_points"].fillna(0).astype(int)
    matches["loser_rank_points"] = matches["loser_rank_points"].fillna(0).astype(int)

    max_rank = matches[["winner_rank", "loser_rank"]].max().max() + 100
    matches["winner_rank"] = matches["winner_rank"].fillna(max_rank).astype(int)
    matches["loser_rank"] = matches["loser_rank"].fillna(max_rank).astype(int)

    matches = impute_missing_match_stats(
        matches,
        stat_columns=[
            "minutes",
            "w_ace",
            "l_ace",
            "w_df",
            "l_df",
            "w_svpt",
            "l_svpt",
            "w_1stIn",
            "l_1stIn",
            "w_1stWon",
            "l_1stWon",
            "w_2ndWon",
            "l_2ndWon",
            "w_SvGms",
            "l_SvGms",
            "w_bpSaved",
            "l_bpSaved",
            "w_bpFaced",
            "l_bpFaced",
        ],
    )

    matches = matches.drop(
        columns=["loser_ioc", "winner_ioc", "winner_name", "loser_name", "score"],
        errors="ignore",
    )
    print("Finished preprocessing ATP matches data.")
    return matches
