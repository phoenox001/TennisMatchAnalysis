# File: src/features/atp_matches.py

import pandas as pd
import numpy as np
import os
import glob

"""
This module contains functions to read and preprocess ATP matches data.
It provides functionality to read all ATP match files from a specified directory,
combine them into a single DataFrame, and perform necessary preprocessing steps
such as handling missing values, renaming columns, and adding match IDs.
It also includes functions to impute missing match statistics based on context
and to preprocess the matches data for further analysis.
"""

def read_all_atp_matches(directory, file_pattern="atp_matches_*.csv"):
    """
    Reads all ATP match files from a directory and concatenates them into a single DataFrame.

    Args:
        directory (str): Directory containing ATP match CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all ATP matches.
    """
    pattern = os.path.join(directory, file_pattern)
    file_list = glob.glob(pattern)

    if not file_list:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in directory '{directory}'."
        )

    atp_matches_list = [pd.read_csv(file) for file in file_list]
    combined_atp_matches = pd.concat(atp_matches_list, ignore_index=True)
    combined_atp_matches["match_id"] = (
        combined_atp_matches["tourney_id"]
        + "-"
        + combined_atp_matches["match_num"].astype("str")
    )
    return combined_atp_matches


def impute_missing_match_stats(
    matches, stat_columns, group_context=["best_of", "surface"]
):
    """
    Imputes missing match statistics using medians grouped by match context
    and adds _missing flags.

    Args:
        matches: (pd.DataFrame) DataFrame containing match statistics.
        stat_columns: list of columns to impute (e.g. ['w_ace', 'l_df', ...])
        group_context: list of context columns (default: ['best_of', 'surface'])

    Returns:
        DataFrame with imputed values and added _missing flags
    """
    matches = matches.copy()

    for col in stat_columns:
        missing_flag = f"{col}_missing"
        matches[missing_flag] = matches[col].isna().astype(int)

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

        matches.drop(columns=[f"{col}_median"], inplace=True)

    return matches


def preprocess_atp_matches_data(matches):
    """
    Cleans ATP matches data by removing duplicates, filling missing data , setting data types, and adding match_id.

    Args:
        matches (pd.DataFrame): DataFrame containing ATP matches.

    Returns:
        pd.DataFrame: Cleaned DataFrame with unique matches and match_id.
    """

    matches = matches.drop_duplicates(subset=["match_id"], keep="first")

    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d")
    matches["winner_is_seeded"] = matches["winner_seed"].notna().astype(int)
    matches["loser_is_seeded"] = matches["loser_seed"].notna().astype(int)
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

    matches["winner_age_missing"] = matches["winner_age"].isna().astype(int)
    matches["loser_age_missing"] = matches["loser_age"].isna().astype(int)
    matches["winner_age"] = (
        matches["winner_age"].fillna(matches["winner_age"].median()).astype(float)
    )
    matches["loser_age"] = (
        matches["loser_age"].fillna(matches["loser_age"].median()).astype(float)
    )

    matches["winner_hand"] = matches["winner_hand"].fillna("U").astype(str)
    matches["loser_hand"] = matches["loser_hand"].fillna("U").astype(str)

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

    return matches
