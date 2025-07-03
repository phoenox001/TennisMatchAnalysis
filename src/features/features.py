# File: src/features/features.py

import pandas as pd
import numpy as np
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
    #     matches (DataFrame): match data
    #     players (DataFrame): player ranking data
    #     role (str): either 'winner' or 'loser'
    # Returns:
    #     DataFrame with enriched match data
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
    players_renamed = players.copy()
    players_renamed = players_renamed.rename(columns=player_cols)

    # Ensure id are of same type
    matches[player_id_col] = matches[player_id_col].astype(
        players_renamed[player_id_col].dtype
    )
    players_renamed[player_id_col] = players_renamed[player_id_col].astype(
        matches[player_id_col].dtype
    )
    matches = matches.sort_values(by=[match_date_col, player_id_col]).reset_index(
        drop=True
    )
    players_renamed = players_renamed.sort_values(
        by=[f"{role}_ranking_date", player_id_col]
    ).reset_index(drop=True)

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
    #
    # Merges player information for both winner and loser roles in the matches DataFrame.

    # Args:
    #     matches (DataFrame): DataFrame containing match data.
    #     players (DataFrame): DataFrame containing player information.
    # Returns:
    #     DataFrame: Matches DataFrame enriched with player information for both winner and loser roles.
    #

    print("Merging player information for both winner and loser roles...")
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

    print("Finished merging player information.")
    return full_information


def compute_historical_averages(matches, player_col, stat_prefix):
    #
    # Computes historical averages for player statistics before each match date.

    # Args:
    #     matches (DataFrame): DataFrame containing match data.
    #     player_col (str): Column name for player IDs (e.g., 'winner_id', 'loser_id').
    #     stat_prefix (str): Prefix for the statistics columns (e.g., 'w_', 'l_').
    # Returns:
    #     DataFrame: Matches DataFrame with added average statistics for each player.
    #

    result = matches.copy()

    cols = [col for col in matches.columns if col.startswith(stat_prefix)]

    for col in cols:
        avg_col = f"{col}_avg"
        result[avg_col] = (
            matches.groupby(player_col)[col]
            .apply(lambda x: x.shift().expanding().mean())
            .reset_index(level=0, drop=True)
        )
    return result


def create_player_match_df(matches):
    #
    # Creates a DataFrame with one row per player per match,
    # including both winner and loser information.

    # Args:
    #     matches (DataFrame): DataFrame containing match data.
    # Returns:
    #     DataFrame: Matches DataFrame with player match history.
    #

    # Prepare winners data
    winners = matches[
        [
            "tourney_date",
            "winner_id",
            "loser_id",
            "surface",
            "tourney_level",
            "tourney_name",
            "winner_rank",
            "loser_rank",
        ]
    ]
    winners = winners.rename(
        columns={
            "winner_id": "player_id",
            "loser_id": "opponent_id",
            "winner_rank": "player_rank",
            "loser_rank": "opponent_rank",
        }
    )
    winners["outcome"] = 1

    # Prepare losers data
    losers = matches[
        [
            "tourney_date",
            "loser_id",
            "winner_id",
            "surface",
            "tourney_level",
            "tourney_name",
            "loser_rank",
            "winner_rank",
        ]
    ]
    losers = losers.rename(
        columns={
            "loser_id": "player_id",
            "winner_id": "opponent_id",
            "loser_rank": "player_rank",
            "winner_rank": "opponent_rank",
        }
    )
    losers["outcome"] = 0

    # Combine and sort
    player_matches = pd.concat([winners, losers], ignore_index=True)
    player_matches = player_matches.sort_values(
        ["player_id", "tourney_date"]
    ).reset_index(drop=True)
    return player_matches


def compute_expanding_feature(data, group_col, date_col, feature_func):
    #
    # General helper to compute an expanding window feature for each group,
    # excluding current row (shifted).
    #

    def apply_func(group):
        group = group.sort_values(date_col)
        # We shift by 1 to exclude current match from aggregation
        return feature_func(group).shift(1)

    return data.groupby(group_col).apply(apply_func).reset_index(level=0, drop=True)


def calc_pct_matches_won(group):
    #
    # Calculate the cumulative percentage of matches won for each player.
    #
    return group["outcome"].expanding().mean()


def calc_avg_opponent_rank_won(group):
    #
    # Calculate the cumulative average rank of opponents beaten by the player.
    # This only considers matches that the player has won.
    #
    won = group.loc[group["outcome"] == 1, "opponent_rank"]
    # For cumulative avg only consider won matches before current match
    cum_avg = won.expanding().mean()
    # We want to align this with full group index, filling NaNs for lost matches
    avg_series = pd.Series(index=group.index, dtype=float)
    avg_series.loc[won.index] = cum_avg
    return avg_series


def calc_avg_tourney_level(group, level_map):
    #
    # Calculate the cumulative average level of tournaments played by the player.
    # The level is mapped to a numeric score based on the provided level_map.
    #
    tourney_scores = group["tourney_level"].map(level_map)
    return tourney_scores.expanding().mean()


def calc_surface_pct(group):
    #
    # Calculate the cumulative percentage of matches played on each surface type.
    # This returns a Series with the percentage of matches played on the current surface
    # at each point in the group.
    # The surface is expected to be a categorical variable with values like 'Hard', 'Clay', 'Grass', etc.
    #
    surface_counts = (
        group["surface"]
        .expanding()
        .apply(
            lambda x: x.value_counts(normalize=True).get(group.iloc[len(x) - 1], 0),
            raw=False,
        )
    )
    return surface_counts


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

    print("Calculating player percentage matches won...")
    # pct matches won
    player_matches["pct_matches_won_prior"] = compute_expanding_feature(
        player_matches, "player_id", "tourney_date", calc_pct_matches_won
    )

    print("Calculating player percentage grand slams won...")
    # pct grand slams won
    grand_slams = player_matches[player_matches["tourney_level"] == "G"]
    pct_gs = compute_expanding_feature(
        grand_slams, "player_id", "tourney_date", calc_pct_matches_won
    )
    player_matches = player_matches.merge(
        pct_gs.rename("pct_grand_slams_won_prior"),
        how="left",
        left_index=True,
        right_index=True,
    )

    print("Calculating player percentage Wimbledon won...")
    # pct Wimbledon won (assuming 'Wimbledon' in tourney_name)
    wimbledon = player_matches[
        player_matches["tourney_name"].str.contains("Wimbledon", case=False, na=False)
    ]
    pct_wimbledon = compute_expanding_feature(
        wimbledon, "player_id", "tourney_date", calc_pct_matches_won
    )
    player_matches = player_matches.merge(
        pct_wimbledon.rename("pct_wimbledon_won_prior"),
        how="left",
        left_index=True,
        right_index=True,
    )

    print("Calculating years on tour...")
    # years on tour
    first_match_year = (
        player_matches.groupby("player_id")["tourney_date"].transform("min").dt.year
    )
    player_matches["years_on_tour"] = (
        player_matches["tourney_date"].dt.year - first_match_year
    )

    print("Calculating average rank of opponents beaten...")
    # average rank of opponents beaten (only won matches)
    def avg_opponent_rank_won(group):
        group = group.sort_values("tourney_date")
        ranks = []
        for i, row in group.iterrows():
            if i == 0:
                ranks.append(np.nan)
                continue
            past = group.iloc[:i]
            beaten_ranks = past.loc[past["outcome"] == 1, "opponent_rank"]
            avg_rank = beaten_ranks.mean() if not beaten_ranks.empty else np.nan
            ranks.append(avg_rank)
        ranks[0] = max(ranks)
        ranks = pd.Series(ranks, index=group.index)
        ranks = ranks.ffill()
        return ranks

    player_matches["avg_opponent_rank_won_prior"] = (
        player_matches.groupby("player_id")
        .apply(avg_opponent_rank_won)
        .reset_index(level=0, drop=True)
    )

    print("Calculating average tournament level...")
    # average level of tourney (numeric score)
    player_matches["avg_tourney_level_prior"] = compute_expanding_feature(
        player_matches,
        "player_id",
        "tourney_date",
        lambda g: calc_avg_tourney_level(g, level_map),
    )

    print("Calculating percentage matches played on each surface...")
    # percentage matches played on surface
    # We calculate a separate percentage per surface type
    for surface_type in ["Hard", "Clay", "Grass", "Carpet"]:
        surface_mask = player_matches["surface"].str.lower() == surface_type.lower()

        def surface_pct_func(g):
            g = g.sort_values("tourney_date")
            return g["surface"].expanding().apply(lambda x: (x == surface_type).mean())

        col_name = f"pct_matches_on_{surface_type.lower()}_prior"
        player_matches[col_name] = compute_expanding_feature(
            player_matches, "player_id", "tourney_date", surface_pct_func
        )

    print("Finished computing player match features.")
    return player_matches


def create_average_match_statistics(matches):
    #
    # Creates average match statistics for each player.
    # The averages are calculated based on past performances before the match date.
    # Args:
    #     matches (DataFrame): DataFrame containing match data.
    # Returns:
    #     DataFrame: Matches DataFrame with added average statistics for each player.
    #

    print("Creating average match statistics for each player...")
    matches = matches.sort_values(by="tourney_date")

    # Calculate averages to winner and loser
    matches = compute_historical_averages(matches, "winner_id", "w_")
    matches = compute_historical_averages(matches, "loser_id", "l_")

    matches = compute_historical_averages(matches, "winner_id", "winner_seed")
    matches = compute_historical_averages(matches, "loser_id", "loser_seed")

    print("Finished creating average match statistics for each player.")
    return matches


def merge_player_features(matches, player_features_df, prefix, player_id_col):
    #
    # Merge player features into matches DataFrame for one player side (winner or loser).

    # Args:
    #     matches: main matches DataFrame
    #     player_features_df: player-match features DataFrame
    #     prefix: string prefix for columns in the output, e.g. 'player1' or 'player2'
    #     player_id_col: column in matches for player id, e.g. 'winner_id' or 'loser_id'
    # Returns:
    #     DataFrame: matches DataFrame with player features merged
    #
    # Select relevant columns from player_features_df to merge
    print(f"Merging player features for {prefix} into matches DataFrame...")
    features_to_merge = player_features_df.drop(
        columns=[
            "opponent_id",
            "outcome",
            "player_rank",
            "opponent_rank",
            "tourney_level",
            "tourney_name",
            "surface",
        ]
    )

    # Rename columns with prefix except player_id and tourney_date
    rename_dict = {
        col: f"{prefix}_{col}"
        for col in features_to_merge.columns
        if col not in ["player_id", "tourney_date"]
    }
    features_to_merge = features_to_merge.rename(columns=rename_dict)

    # Merge on player_id and tourney_date
    merged = matches.merge(
        features_to_merge,
        how="left",
        left_on=[player_id_col, "tourney_date"],
        right_on=["player_id", "tourney_date"],
    )

    # Drop duplicated columns from merge
    merged = merged.drop(columns=["player_id"])

    print(f"Finished merging player features for {prefix}.")
    return merged


def randomize_player_order(matches):
    #
    # Randomly flips player1 and player2 columns in the DataFrame.
    # This is useful for creating a balanced dataset where the order of players does not bias the model.

    # Args:
    #     matches (DataFrame): DataFrame containing match data with player1 and player2 columns.
    # Returns:
    #     DataFrame: DataFrame with player1 and player2 columns randomly flipped.
    #

    print("Randomizing player order in matches DataFrame...")
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

    print("Finished randomizing player order.")
    return matches.drop(columns=["winner"])


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
    matches = merge_all_player_info(matches, players)
    matches = create_average_match_statistics(matches)
    player_features = compute_features(matches)
    matches = merge_player_features(matches, player_features, "player1", "winner_id")
    matches = merge_player_features(
        matches, player_features, prefix="player2", player_id_col="loser_id"
    )
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

    print("Finished creating feature DataFrame.")
    return matches_reduced
