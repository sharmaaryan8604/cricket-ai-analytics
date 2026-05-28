import pandas as pd
import streamlit as st

from utils.prediction import load_ball_data, load_model


def mode_or_first(series):

    cleaned = series.dropna()

    if cleaned.empty:

        return ""

    modes = cleaned.mode()

    if not modes.empty:

        return modes.iloc[0]

    return cleaned.iloc[0]


def prepare_ball_dataframe(ball_df):

    enriched_df = ball_df.copy()
    enriched_df["date_parsed"] = pd.to_datetime(
        enriched_df["date"],
        errors="coerce",
    )
    enriched_df["season"] = enriched_df["date_parsed"].dt.year
    enriched_df["legal_ball"] = (
        enriched_df["legal_ball"]
        .fillna(0)
        .astype(int)
    )
    enriched_df["ball_faced"] = (
        enriched_df["wides"]
        .fillna(0)
        .eq(0)
        .astype(int)
    )
    enriched_df["is_boundary"] = (
        enriched_df["batter_runs"]
        .isin([4, 6])
        .astype(int)
    )
    enriched_df["bowler_runs"] = (
        enriched_df["total_runs"] -
        enriched_df["byes"].fillna(0) -
        enriched_df["legbyes"].fillna(0)
    )

    previous_bowler_wickets = (
        enriched_df.groupby(
            ["match_id", "innings", "bowler"]
        )["bowler_wickets"]
        .shift(fill_value=0)
    )

    enriched_df["wicket_credit"] = (
        enriched_df["bowler_wickets"] - previous_bowler_wickets
    ).clip(lower=0)

    return enriched_df


@st.cache_resource
def get_model():

    return load_model()


@st.cache_data
def get_ball_df():

    return prepare_ball_dataframe(load_ball_data())


@st.cache_data
def get_match_summary_df():

    ball_df = get_ball_df()

    return (
        ball_df[
            [
                "match_id",
                "season",
                "date_parsed",
                "winner",
                "city",
                "venue",
            ]
        ]
        .drop_duplicates(subset=["match_id"])
        .sort_values(["season", "date_parsed", "match_id"])
        .reset_index(drop=True)
    )


@st.cache_data
def get_batting_leaders():

    ball_df = get_ball_df()

    batting_df = (
        ball_df.groupby(["season", "batter"], as_index=False)
        .agg(
            team=("batting_team", mode_or_first),
            runs=("batter_runs", "sum"),
            balls=("ball_faced", "sum"),
            fours=("batter_runs", lambda values: int((values == 4).sum())),
            sixes=("batter_runs", lambda values: int((values == 6).sum())),
            matches=("match_id", "nunique"),
        )
        .rename(columns={"batter": "player"})
    )

    batting_df = batting_df[batting_df["runs"] > 0].copy()
    batting_df["strike_rate"] = (
        batting_df["runs"] * 100 / batting_df["balls"]
    ).where(
        batting_df["balls"] > 0,
        0,
    ).round(2)

    batting_df = batting_df.sort_values(
        ["season", "runs", "strike_rate", "fours", "player"],
        ascending=[True, False, False, False, True],
    )
    batting_df["rank"] = (
        batting_df.groupby("season")
        .cumcount() + 1
    )

    return batting_df.reset_index(drop=True)


def format_bowling_overs(balls_bowled):

    completed_overs = int(balls_bowled) // 6
    over_balls = int(balls_bowled) % 6

    return f"{completed_overs}.{over_balls}"


@st.cache_data
def get_bowling_leaders():

    ball_df = get_ball_df()

    bowling_df = (
        ball_df.groupby(["season", "bowler"], as_index=False)
        .agg(
            team=("bowling_team", mode_or_first),
            wickets=("wicket_credit", "sum"),
            balls=("legal_ball", "sum"),
            runs_conceded=("bowler_runs", "sum"),
            matches=("match_id", "nunique"),
        )
        .rename(columns={"bowler": "player"})
    )

    bowling_df = bowling_df[bowling_df["wickets"] > 0].copy()
    bowling_df["economy"] = (
        bowling_df["runs_conceded"] * 6 / bowling_df["balls"]
    ).where(
        bowling_df["balls"] > 0,
        0,
    ).round(2)
    bowling_df["strike_rate"] = (
        bowling_df["balls"] / bowling_df["wickets"]
    ).where(
        bowling_df["wickets"] > 0,
        0,
    ).round(2)
    bowling_df["overs"] = bowling_df["balls"].apply(format_bowling_overs)

    bowling_df = bowling_df.sort_values(
        ["season", "wickets", "economy", "strike_rate", "player"],
        ascending=[True, False, True, True, True],
    )
    bowling_df["rank"] = (
        bowling_df.groupby("season")
        .cumcount() + 1
    )

    return bowling_df.reset_index(drop=True)


@st.cache_data
def get_season_caps():

    batting_df = get_batting_leaders()
    bowling_df = get_bowling_leaders()

    orange_cap_df = (
        batting_df[batting_df["rank"] == 1][
            ["season", "player", "team", "runs", "strike_rate", "matches"]
        ]
        .rename(columns={
            "player": "orange_cap",
            "team": "orange_team",
            "runs": "orange_runs",
            "strike_rate": "orange_strike_rate",
            "matches": "orange_matches",
        })
    )

    purple_cap_df = (
        bowling_df[bowling_df["rank"] == 1][
            ["season", "player", "team", "wickets", "economy", "matches"]
        ]
        .rename(columns={
            "player": "purple_cap",
            "team": "purple_team",
            "wickets": "purple_wickets",
            "economy": "purple_economy",
            "matches": "purple_matches",
        })
    )

    caps_df = orange_cap_df.merge(
        purple_cap_df,
        on="season",
        how="outer",
    )

    return caps_df.sort_values("season").reset_index(drop=True)


@st.cache_data
def get_season_overview():

    ball_df = get_ball_df()
    match_df = get_match_summary_df()

    season_df = (
        ball_df.groupby("season", as_index=False)
        .agg(
            total_runs=("total_runs", "sum"),
            total_wickets=("wicket", "sum"),
            teams=("batting_team", "nunique"),
            venues=("venue", "nunique"),
            cities=("city", "nunique"),
        )
        .merge(
            match_df.groupby("season", as_index=False).agg(
                matches=("match_id", "nunique")
            ),
            on="season",
            how="left",
        )
        .sort_values("season")
        .reset_index(drop=True)
    )

    return season_df


def latest_season():

    season_df = get_season_overview()

    return int(season_df["season"].max())
