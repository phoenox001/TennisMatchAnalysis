# features.py

import pandas as pd
import numpy as np
import os
import glob


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
    return combined_atp_matches


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
