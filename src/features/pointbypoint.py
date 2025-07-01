# File: src/features/pointbypoint.py

import pandas as pd
import numpy as np
import os
import glob


def read_all_pointbypoint_data(directory, file_pattern="*.csv"):
    """
    Reads all point-by-point data files from a directory and concatenates them into a single DataFrame.

    Args:
        directory (str): Directory containing point-by-point CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all point-by-point data.
    """
    pattern = os.path.join(directory, file_pattern)
    file_list = glob.glob(pattern)

    if not file_list:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in directory '{directory}'."
        )

    pointbypoint_data_list = [pd.read_csv(file) for file in file_list]
    combined_pointbypoint_data = pd.concat(pointbypoint_data_list, ignore_index=True)
    return combined_pointbypoint_data
