from pathlib import Path
import pickle

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "xgboost_win_predictor.pkl"
BALL_DATA_PATH = BASE_DIR / "data" / "ipl_ball_by_ball.csv"


TEAMS = sorted([
    "Sunrisers Hyderabad",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Kings XI Punjab",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Delhi Daredevils",
])


TEAM_LOGOS = {
    "Chennai Super Kings": BASE_DIR / "images" / "logos" / "CSK.png",
    "Mumbai Indians": BASE_DIR / "images" / "logos" / "MI.png",
    "Royal Challengers Bangalore": BASE_DIR / "images" / "logos" / "RCB.png",
    "Kolkata Knight Riders": BASE_DIR / "images" / "logos" / "KKR.png",
    "Rajasthan Royals": BASE_DIR / "images" / "logos" / "RR.png",
    "Sunrisers Hyderabad": BASE_DIR / "images" / "logos" / "SRH.png",
    "Kings XI Punjab": BASE_DIR / "images" / "logos" / "PBKS.png",
    "Punjab Kings": BASE_DIR / "images" / "logos" / "PBKS.png",
    "Delhi Daredevils": BASE_DIR / "images" / "logos" / "DC.png",
    "Delhi Capitals": BASE_DIR / "images" / "logos" / "DC.png",
    "Deccan Chargers": BASE_DIR / "images" / "logos" / "DCG.png",
    "Lucknow Super Giants": BASE_DIR / "images" / "logos" / "LSG.png",
    "Gujarat Titans": BASE_DIR / "images" / "logos" / "GT.png",
    "Pune Warriors": BASE_DIR / "images" / "logos" / "PW.png",
    "Rising Pune Supergiant": BASE_DIR / "images" / "logos" / "RPS.png",
    "Rising Pune Supergiants": BASE_DIR / "images" / "logos" / "RPS.png",
    "Gujarat Lions": BASE_DIR / "images" / "logos" / "GL.png",
    "Kochi Tuskers Kerala": BASE_DIR / "images" / "logos" / "KTK.png",
}


TEAM_COLORS = {
    "Chennai Super Kings": "#fdb913",
    "Mumbai Indians": "#004ba0",
    "Royal Challengers Bangalore": "#d71920",
    "Kolkata Knight Riders": "#3a225d",
    "Rajasthan Royals": "#ea1a85",
    "Sunrisers Hyderabad": "#ff822a",
    "Kings XI Punjab": "#dd1f2d",
    "Punjab Kings": "#dd1f2d",
    "Delhi Daredevils": "#004c93",
    "Delhi Capitals": "#004c93",
    "Lucknow Super Giants": "#00a0e3",
    "Gujarat Titans": "#1b2133",
}


CITIES = sorted([
    "Hyderabad",
    "Bangalore",
    "Mumbai",
    "Kolkata",
    "Chennai",
    "Delhi",
    "Jaipur",
    "Pune",
    "Abu Dhabi",
    "Sharjah",
])


def load_model():

    with open(MODEL_PATH, "rb") as model_file:

        return pickle.load(model_file)


def load_ball_data():

    return pd.read_csv(BALL_DATA_PATH)


def build_valid_overs():

    valid_overs = []

    for over in range(20):

        for ball in range(6):

            valid_overs.append(f"{over}.{ball}")

    valid_overs.append("20")

    return valid_overs


def clamp_value(value, lower, upper):

    return max(
        lower,
        min(value, upper),
    )


def compute_momentum_index(
    current_run_rate,
    required_run_rate,
    wickets_remaining,
    last_30_runs,
    last_30_dot_balls,
    last_30_boundaries,
):

    run_rate_edge = (
        current_run_rate - required_run_rate
    ) * 12

    wicket_buffer = (
        wickets_remaining - 5
    ) * 4

    recent_scoring = (
        last_30_runs - 30
    ) * 1.5

    boundary_boost = last_30_boundaries * 3
    dot_ball_drag = last_30_dot_balls * 2.5

    momentum_index = (
        run_rate_edge +
        wicket_buffer +
        recent_scoring +
        boundary_boost -
        dot_ball_drag
    )

    return round(
        clamp_value(
            momentum_index,
            -100,
            100,
        ),
        1,
    )


def get_momentum_label(momentum_index):

    if momentum_index >= 45:

        return "Strong batting momentum"

    if momentum_index >= 15:

        return "Batting side on top"

    if momentum_index > -15:

        return "Momentum balanced"

    if momentum_index > -45:

        return "Bowling side applying pressure"

    return "Bowling side dominating"


def get_team_color(team_name, fallback="#00bfff"):

    return TEAM_COLORS.get(
        team_name,
        fallback,
    )


def format_delta(value):

    return f"{value:+.2f}"


def calculate_match_context(target, score, overs, wickets):

    over_part = int(overs)
    ball_part = int(round((overs - over_part) * 10))
    balls_completed = (over_part * 6) + ball_part

    runs_remaining = target - score
    balls_remaining = 120 - balls_completed
    wickets_remaining = 10 - wickets

    current_run_rate = (
        score / overs
        if overs > 0 else 0
    )

    required_run_rate = (
        (runs_remaining * 6) / balls_remaining
        if balls_remaining > 0 else 0
    )

    last_30_runs = current_run_rate * 5
    last_30_wickets = wickets
    last_30_dot_balls = max(0, int(15 - current_run_rate))
    last_30_boundaries = max(1, int(current_run_rate / 2))

    if overs <= 6:

        phase = "Powerplay"

    elif overs <= 16:

        phase = "Middle"

    else:

        phase = "Death"

    return {
        "balls_completed": balls_completed,
        "runs_remaining": runs_remaining,
        "balls_remaining": balls_remaining,
        "wickets_remaining": wickets_remaining,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,
        "last_30_runs": last_30_runs,
        "last_30_wickets": last_30_wickets,
        "last_30_dot_balls": last_30_dot_balls,
        "last_30_boundaries": last_30_boundaries,
        "phase": phase,
    }


def build_prediction_input(
    batting_team,
    bowling_team,
    city,
    context,
):

    return pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "city": [city],
        "phase": [context["phase"]],
        "runs_remaining": [context["runs_remaining"]],
        "balls_remaining": [context["balls_remaining"]],
        "wickets_remaining": [context["wickets_remaining"]],
        "current_run_rate": [context["current_run_rate"]],
        "required_run_rate": [context["required_run_rate"]],
        "last_30_runs": [context["last_30_runs"]],
        "last_30_wickets": [context["last_30_wickets"]],
        "last_30_dot_balls": [context["last_30_dot_balls"]],
        "last_30_boundaries": [context["last_30_boundaries"]],
    })
