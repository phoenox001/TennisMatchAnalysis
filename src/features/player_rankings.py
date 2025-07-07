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

    return pd.read_csv(
        file_path,
        dtype={"player_id": "int64", "dob": "str", "height": "float64"},
    )


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

    atp_rankings_list = [
        pd.read_csv(file, dtype={"player": "int64", "points": "float64"})
        for file in file_list
    ]
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
    players["dob"] = pd.to_datetime(players["dob"], format="%Y%m%d", errors="coerce")
    players["player_id"] = players["player_id"].astype("int64")
    rankings["player_id"] = players["player_id"].astype("int64")
    ranked_players = pd.merge(rankings, players, on="player_id", how="left")
    ranked_players = ranked_players.sort_values(
        by=["ranking_date", "rank"], ascending=[False, True]
    )
    return ranked_players


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

    players = players[pd.notna(players["name_first"])]
    players = players[pd.notna(players["name_last"])]

    players["points"] = players["points"].fillna(0).astype(int)

    # players.loc["height"] = players["height"].fillna(players["height"].median())
    players["hand"] = players["hand"].fillna("U")
    players["ranking_date"] = pd.to_datetime(
        players["ranking_date"], format="%Y%m%d", errors="coerce"
    )

    print(players.columns)

    players = players.drop(
        columns=["wikidata_id", "dob", "height", "ioc"], errors="ignore"
    )
    players["player_id"] = players["player_id"].astype("int64")
    players = players.sort_values(by="player_id")

    print("Finished preprocessing player data.")
    return players
