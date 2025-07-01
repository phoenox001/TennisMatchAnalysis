# File: src/features/player_rankings.py

import pandas as pd
import numpy as np
import os
import glob


def read_atp_players(file_path="../data/atp_matches/atp_players.csv"):
    """
    Reads ATP players data from a CSV file.

    Args:
        file_path (str): Path to the ATP players CSV file.

    Returns:
        pd.DataFrame: DataFrame containing ATP players data.
    """
    return pd.read_csv(file_path)


def read_all_atp_rankings(directory, file_pattern="atp_rankings_*.csv"):
    """
    Reads all ATP rankings files from a directory and concatenates them into a single DataFrame.

    Args:
        directory (str): Directory containing ATP rankings CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all ATP rankings.
    """
    pattern = os.path.join(directory, file_pattern)
    file_list = glob.glob(pattern)

    if not file_list:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in directory '{directory}'."
        )

    atp_rankings_list = [pd.read_csv(file) for file in file_list]
    combined_atp_rankings = pd.concat(atp_rankings_list, ignore_index=True)
    return combined_atp_rankings


def combine_rankings_with_players(rankings, players):
    """
    Combines ATP rankings with player information.

    Args:
        rankings (pd.DataFrame): DataFrame containing ATP rankings.
        players (pd.DataFrame): DataFrame containing player information.

    Returns:
        pd.DataFrame: Combined DataFrame with rankings and player information.
    """
    rankings = rankings.rename(columns={"player": "player_id"})
    ranked_players = pd.merge(rankings, players, on="player_id", how="left")
    ranked_players = ranked_players.sort_values(
        by=["ranking_date", "rank"], ascending=[False, True]
    )
    return ranked_players


def calculate_missing_dob(player_id, matches):
    """
    Calculates missing date of birth (dob) for players based on their first match date.
    We assume a debut age of 18 years.

    Args:
        players (pd.DataFrame): DataFrame containing player information.
        matches (pd.DataFrame): DataFrame containing match information.

    Returns:
        pd.DataFrame: Updated DataFrame with calculated dob for players.
    """
    # Get the first match date for the player
    first_match_date_winner = (
        matches[matches["player_id"] == player_id]
        .groupby("winner_id")["date"]
        .min()
        .reset_index()
    )
    first_match_date_loser = (
        matches[matches["player_id"] == player_id]
        .groupby("loser_id")["date"]
        .min()
        .reset_index()
    )
    first_match_dates = pd.concat(
        [first_match_date_winner, first_match_date_loser],
        ignore_index=True,
    )
    first_match_dates = first_match_dates.drop_duplicates(subset=["date"])

    first_match_dates["dob"] = first_match_dates["date"] - pd.DateOffset(years=18)
    first_match_dates = first_match_dates.rename(columns={"date": "dob"})

    # Fill missing dob values with calculated values
    players["dob"] = players["dob"].fillna(players["dob_y"])
    players = players.drop(columns=["dob_y"], errors="ignore")

    return players


def clean_player_data(players):
    """
    Cleans player data by removing duplicates, and Missing values, setting data types and sorting.

    Args:
        players (pd.DataFrame): DataFrame containing player information.

    Returns:
        pd.DataFrame: Cleaned DataFrame with unique players sorted by player_id.
    """
    # remove duplicates based on player_id
    players = players.drop_duplicates(subset=["player_id"], keep="first")

    # set data types and fill missing values
    players = players["height"].fillna(players["height"].mean())
    players["dob"] = pd.to_datetime(players["dob"].astype("str"), format="%Y%m%d")
    players["ioc"] = players["ioc"].fillna("Unknown")
    players["ioc"] = players["ioc"].str.upper()
    players["hand"] = players["hand"].fillna("U")

    # drop players with no name
    players = players[players["name_first"].notna()]
    players = players[players["name_last"].notna()]

    # drop unused columns
    players = players.drop(columns="wikidata_id", errors="ignore")
    players = players.sort_values(by="player_id")
    return players
