# File: src/features/player_rankings.py

import pandas as pd
import numpy as np
import os
import glob

#
# This module contains functions to read and preprocess ATP player rankings data.
# It provides functionality to read all ATP player files from a specified directory,
# combine them into a single DataFrame, and perform necessary preprocessing steps
# such as handling missing values, renaming columns, and adding player information.
# It also includes functions to read ATP players data and combine rankings with player information.
#


def read_atp_players(file_path):
    #
    # Reads ATP players data from a CSV file.

    # Args:
    #     file_path (str): Path to the ATP players CSV file.

    # Returns:
    #     pd.DataFrame: DataFrame containing ATP players data.
    #

    print(f"Reading ATP players data from: {file_path}")

    return pd.read_csv(file_path)


def read_all_atp_rankings(directory, file_pattern="atp_rankings_*.csv"):
    #
    # Reads all ATP rankings files from a directory and concatenates them into a single DataFrame.

    # Args:
    #     directory (str): Directory containing ATP rankings CSV files.

    # Returns:
    #     pd.DataFrame: Concatenated DataFrame of all ATP rankings.
    #

    print(f"Reading ATP rankings from directory: {directory}")

    pattern = os.path.join(directory, file_pattern)
    file_list = glob.glob(pattern)

    if not file_list:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in directory '{directory}'."
        )

    atp_rankings_list = [pd.read_csv(file) for file in file_list]
    combined_atp_rankings = pd.concat(atp_rankings_list, ignore_index=True)
    combined_atp_rankings["ranking_date"] = pd.to_datetime(
        combined_atp_rankings["ranking_date"], format="%Y%m%d"
    )
    return combined_atp_rankings


def combine_rankings_with_players(rankings, players):
    #
    # Combines ATP rankings with player information.

    # Args:
    #     rankings (pd.DataFrame): DataFrame containing ATP rankings.
    #     players (pd.DataFrame): DataFrame containing player information.

    # Returns:
    #     pd.DataFrame: Combined DataFrame with rankings and player information.
    #

    print("Combining ATP rankings with player information...")

    rankings = rankings.rename(columns={"player": "player_id"})
    ranked_players = pd.merge(rankings, players, on="player_id", how="left")
    ranked_players = ranked_players.sort_values(
        by=["ranking_date", "rank"], ascending=[False, True]
    )
    return ranked_players


def calculate_missing_dob(players, matches):
    #
    # Calculates missing date of birth (dob) for players based on their first match date.
    # We assume a debut age of 18 years.

    # Args:
    #     players (pd.DataFrame): DataFrame containing player information with missing dob.
    #     matches (pd.DataFrame): DataFrame containing match history.

    # Returns:
    #     pd.Datetime: The estimated dob of the player.
    #

    print("Calculating missing date of birth (dob) for players...")

    matches = matches.sort_values(by="tourney_date")

    first_match_dates = matches.groupby("player_id")["tourney_date"].min()

    # Calculate dob assuming a debut age of 18 years
    players["dob"] = players.apply(
        lambda row: (
            row["dob"]
            if pd.notna(row["dob"])
            else first_match_dates.get(row["player_id"], pd.NaT)
            - pd.DateOffset(years=18)
        ),
        axis=1,
    )

    return players


def preprocess_player_data(players):
    #
    # Cleans player data by removing duplicates, and Missing values, setting data types and sorting.

    # Args:
    #     players (pd.DataFrame): DataFrame containing player information.

    # Returns:
    #     pd.DataFrame: Cleaned DataFrame with unique players sorted by player_id.
    #

    print("Preprocessing player data...")
    players = players.copy()
    players = players.drop_duplicates(subset=["player_id"], keep="first")

    players = players[players["name_first"].notna()]
    players = players[players["name_last"].notna()]

    # players.loc["height"] = players["height"].fillna(players["height"].median())
    players["ioc"] = players["ioc"].str.upper()
    players["ioc"] = players["ioc"].fillna("Unknown")
    players["hand"] = players["hand"].fillna("U")
    players["ranking_date"] = pd.to_datetime(
        players["ranking_date"], format="%Y%m%d", errors="coerce"
    )
    players["points_missing"] = players["points"].isna().astype(int)
    players["points"] = players["points"].fillna(0).astype(int)

    print(players.columns)

    players = players.drop(columns=["wikidata_id", "dob", "height"], errors="ignore")
    players["player_id"] = players["player_id"].astype("int64", errors="ignore")
    players = players.sort_values(by="player_id")

    print("Finished preprocessing player data.")
    return players
