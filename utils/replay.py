import pandas as pd

from utils.prediction import (
    build_prediction_input,
    format_delta,
    get_team_color,
)


def prepare_match_options(ball_df):

    match_options = (
        ball_df[
            ["match_id", "date", "batting_team", "bowling_team"]
        ]
        .drop_duplicates(subset=["match_id"])
        .copy()
    )

    match_options["formatted_date"] = pd.to_datetime(
        match_options["date"],
        errors="coerce",
    ).dt.strftime("%d %b %Y")
    match_options["season"] = pd.to_datetime(
        match_options["date"],
        errors="coerce",
    ).dt.year

    match_options["formatted_date"] = (
        match_options["formatted_date"]
        .fillna(match_options["date"].astype(str))
    )

    match_options["match_label"] = (
        match_options["formatted_date"] +
        " - " +
        match_options["batting_team"] +
        " vs " +
        match_options["bowling_team"]
    )

    return match_options


def filter_match_replay(ball_df, selected_match):

    return (
        ball_df[
            (ball_df["match_id"] == selected_match) &
            (ball_df["innings"] == 2)
        ]
        .copy()
        .sort_values("balls_elapsed")
        .reset_index(drop=True)
    )


def format_over_ball(balls_elapsed):

    balls = int(balls_elapsed)
    over_num = balls // 6
    ball_num = balls % 6

    if ball_num == 0 and balls != 0:

        return f"{over_num}"

    return f"{over_num}.{ball_num}"


def generate_replay_data(match_df, model):

    probabilities = []
    overs_progress = []
    event_x = []
    event_y = []
    event_text = []
    momentum_shifts = []
    previous_probability = None

    for _, row in match_df.iterrows():

        try:

            input_df = build_prediction_input(
                batting_team=row["batting_team"],
                bowling_team=row["bowling_team"],
                city=row["city"],
                context={
                    "phase": row["phase"],
                    "runs_remaining": row["runs_remaining"],
                    "balls_remaining": row["balls_remaining"],
                    "wickets_remaining": row["wickets_remaining"],
                    "current_run_rate": row["current_run_rate"],
                    "required_run_rate": row["required_run_rate"],
                    "last_30_runs": row["last_30_runs"],
                    "last_30_wickets": row["last_30_wickets"],
                    "last_30_dot_balls": row["last_30_dot_balls"],
                    "last_30_boundaries": row["last_30_boundaries"],
                },
            )

            probability = model.predict_proba(input_df)[0][1] * 100
            probabilities.append(probability)

            if previous_probability is None:

                momentum_shift = 0.0

            else:

                momentum_shift = round(
                    probability - previous_probability,
                    2,
                )

            momentum_shifts.append(momentum_shift)
            previous_probability = probability

            current_ball = format_over_ball(row["balls_elapsed"])
            overs_progress.append(current_ball)

            if row["wicket"] == 1:

                event_x.append(current_ball)
                event_y.append(probability)
                event_text.append("WICKET")

            elif row["batter_runs"] == 6:

                event_x.append(current_ball)
                event_y.append(probability)
                event_text.append("SIX")

            elif row["batter_runs"] == 4:

                event_x.append(current_ball)
                event_y.append(probability)
                event_text.append("FOUR")

        except Exception:

            continue

    if not probabilities:

        return None

    replay_df = match_df.iloc[:len(probabilities)].copy()
    replay_df["over_ball"] = overs_progress
    replay_df["win_probability"] = probabilities
    replay_df["momentum_shift"] = momentum_shifts

    replay_batting_team = replay_df["batting_team"].iloc[0]
    replay_bowling_team = replay_df["bowling_team"].iloc[0]

    replay_batting_color = get_team_color(replay_batting_team)
    replay_bowling_color = get_team_color(
        replay_bowling_team,
        "#dc2626",
    )

    replay_df["rolling_momentum"] = (
        replay_df["momentum_shift"]
        .rolling(6, min_periods=1)
        .mean()
        .round(2)
    )

    replay_df["momentum_color"] = replay_df[
        "momentum_shift"
    ].apply(
        lambda value: (
            replay_batting_color
            if value >= 0 else replay_bowling_color
        )
    )

    biggest_swing_index = replay_df["momentum_shift"].abs().idxmax()
    biggest_swing_row = replay_df.loc[biggest_swing_index]

    net_momentum = round(
        replay_df["momentum_shift"].sum(),
        2,
    )

    batting_push_balls = int(
        (replay_df["momentum_shift"] > 0).sum()
    )

    bowling_push_balls = int(
        (replay_df["momentum_shift"] < 0).sum()
    )

    swing_team = (
        biggest_swing_row["batting_team"]
        if biggest_swing_row["momentum_shift"] >= 0
        else biggest_swing_row["bowling_team"]
    )

    return {
        "probabilities": probabilities,
        "overs_progress": overs_progress,
        "event_x": event_x,
        "event_y": event_y,
        "event_text": event_text,
        "replay_df": replay_df,
        "replay_batting_team": replay_batting_team,
        "replay_bowling_team": replay_bowling_team,
        "replay_batting_color": replay_batting_color,
        "replay_bowling_color": replay_bowling_color,
        "biggest_swing_row": biggest_swing_row,
        "net_momentum_display": format_delta(net_momentum),
        "biggest_swing_display": format_delta(
            biggest_swing_row["momentum_shift"]
        ),
        "net_momentum": net_momentum,
        "batting_push_balls": batting_push_balls,
        "bowling_push_balls": bowling_push_balls,
        "swing_team": swing_team,
    }
