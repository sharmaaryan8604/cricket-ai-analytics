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
}


TEAM_COLORS = {
    "Chennai Super Kings": "#fdb913",
    "Mumbai Indians": "#004ba0",
    "Royal Challengers Bangalore": "#d71920",
    "Royal Challengers Bengaluru": "#d71920",
    "Kolkata Knight Riders": "#3a225d",
    "Rajasthan Royals": "#ea1a85",
    "Sunrisers Hyderabad": "#ff822a",
    "Kings XI Punjab": "#dd1f2d",
    "Punjab Kings": "#dd1f2d",
    "Delhi Daredevils": "#004c93",
    "Delhi Capitals": "#004c93",
    "Lucknow Super Giants": "#00a0e3",
    "Gujarat Titans": "#1b2133",
    "Deccan Chargers": "#1e3a8a",
    "Gujarat Lions": "#f97316",
    "Pune Warriors": "#22c55e",
    "Rising Pune Supergiant": "#7c3aed",
    "Rising Pune Supergiants": "#7c3aed",
    "Kochi Tuskers Kerala": "#16a34a",
}


TEAM_SHORT_NAMES = {
    "Chennai Super Kings": "CSK",
    "Mumbai Indians": "MI",
    "Royal Challengers Bangalore": "RCB",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",
    "Kings XI Punjab": "PBKS",
    "Punjab Kings": "PBKS",
    "Delhi Daredevils": "DC",
    "Delhi Capitals": "DC",
    "Lucknow Super Giants": "LSG",
    "Gujarat Titans": "GT",
    "Deccan Chargers": "DEC",
    "Gujarat Lions": "GL",
    "Pune Warriors": "PW",
    "Rising Pune Supergiant": "RPS",
    "Rising Pune Supergiants": "RPS",
    "Kochi Tuskers Kerala": "KTK",
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


def get_available_teams(ball_df):

    teams = set(ball_df["batting_team"].dropna())
    teams.update(ball_df["bowling_team"].dropna())

    return sorted(teams)


def get_available_cities(ball_df):

    return sorted(ball_df["city"].dropna().unique().tolist())


def build_valid_overs():

    valid_overs = []

    for over in range(20):

        for ball in range(6):

            valid_overs.append(f"{over}.{ball}")

    valid_overs.append("20")

    return valid_overs


def overs_to_balls(overs):

    over_part = int(overs)
    ball_part = int(round((overs - over_part) * 10))

    return (over_part * 6) + ball_part


def balls_to_overs_float(balls_completed):

    if balls_completed <= 0:

        return 0.0

    completed_overs = balls_completed // 6
    ball_part = balls_completed % 6

    return float(f"{completed_overs}.{ball_part}")


def format_balls_as_overs(balls_completed):

    completed_overs = int(balls_completed) // 6
    ball_part = int(balls_completed) % 6

    if ball_part == 0:

        return f"{completed_overs}.0"

    return f"{completed_overs}.{ball_part}"


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


def get_team_short_name(team_name):

    if team_name in TEAM_SHORT_NAMES:

        return TEAM_SHORT_NAMES[team_name]

    initials = [
        word[0]
        for word in str(team_name).split()
        if word and word[0].isalnum()
    ]

    return "".join(initials[:4]).upper() or "IPL"


def get_team_logo_path(team_name):

    logo_path = TEAM_LOGOS.get(team_name)

    if logo_path and logo_path.exists():

        return logo_path

    return None


def format_delta(value):

    return f"{value:+.2f}"


def derive_recent_phase_stats(recent_events):

    if not recent_events:

        return None

    legal_events = []

    for event in reversed(recent_events):

        if event.get("legal_ball", 1):

            legal_events.append(event)

        if len(legal_events) == 30:

            break

    if not legal_events:

        return None

    legal_events.reverse()

    last_30_runs = sum(
        event.get("runs", 0)
        for event in legal_events
    )

    last_30_wickets = sum(
        event.get("wicket", 0)
        for event in legal_events
    )

    last_30_dot_balls = sum(
        1
        for event in legal_events
        if event.get("runs", 0) == 0 and not event.get("wicket", 0)
    )

    last_30_boundaries = sum(
        1
        for event in legal_events
        if event.get("is_boundary", False)
    )

    return {
        "last_30_runs": last_30_runs,
        "last_30_wickets": last_30_wickets,
        "last_30_dot_balls": last_30_dot_balls,
        "last_30_boundaries": last_30_boundaries,
    }


def calculate_match_context(
    target,
    score,
    overs,
    wickets,
    recent_events=None,
):

    balls_completed = overs_to_balls(overs)
    runs_remaining = max(target - score, 0)
    balls_remaining = max(120 - balls_completed, 0)
    wickets_remaining = max(10 - wickets, 0)

    current_run_rate = (
        score / overs
        if overs > 0 else 0
    )

    required_run_rate = (
        (runs_remaining * 6) / balls_remaining
        if balls_remaining > 0 else 0
    )

    recent_stats = derive_recent_phase_stats(recent_events)

    if recent_stats:

        last_30_runs = recent_stats["last_30_runs"]
        last_30_wickets = recent_stats["last_30_wickets"]
        last_30_dot_balls = recent_stats["last_30_dot_balls"]
        last_30_boundaries = recent_stats["last_30_boundaries"]

    else:

        last_30_runs = round(current_run_rate * 5, 1)
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


def predict_win_probabilities(model, input_df, context):

    if context["runs_remaining"] <= 0:

        return 0.0, 1.0

    if (
        context["balls_remaining"] <= 0 or
        context["wickets_remaining"] <= 0
    ):

        return 1.0, 0.0

    result = model.predict_proba(input_df)

    return float(result[0][0]), float(result[0][1])
