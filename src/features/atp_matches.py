# File: src/features/atp_matches.py

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
    combined_atp_matches["match_id"] = (
        combined_atp_matches["tourney_id"]
        + "-"
        + combined_atp_matches["match_num"].astype("str")
    )
    return combined_atp_matches


def clean_atp_matches(matches):
    """
    Cleans ATP matches data by removing duplicates, setting data types, and adding match_id.

    Args:
        matches (pd.DataFrame): DataFrame containing ATP matches.

    Returns:
        pd.DataFrame: Cleaned DataFrame with unique matches and match_id.
    """
    # remove duplicates based on match_id
    matches = matches.drop_duplicates(subset=["match_id"], keep="first")

    # set data types
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d")

    # add match_id
    matches["match_id"] = (
        matches["tourney_id"] + "-" + matches["match_num"].astype("str")
    )

    return matches
