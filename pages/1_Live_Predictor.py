import pandas as pd
import shap
import streamlit as st

from utils.charts import (
    create_live_match_figure,
    create_shap_waterfall_figure,
)
from utils.commentary import (
    build_ai_insights,
    build_momentum_drivers,
)
from utils.dashboard import (
    get_ball_df,
    get_model,
)
from utils.prediction import (
    balls_to_overs_float,
    build_prediction_input,
    calculate_match_context,
    compute_momentum_index,
    format_balls_as_overs,
    get_available_cities,
    get_available_teams,
    get_momentum_label,
    get_team_color,
    predict_win_probabilities,
)
from utils.ui import (
    apply_app_styles,
    render_metric_card,
    render_page_header,
    render_team_pill,
)


st.set_page_config(
    page_title="Live IPL Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styles()

ball_df = get_ball_df()
model = get_model()
preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["classifier"]

teams = get_available_teams(ball_df)
cities = get_available_cities(ball_df)


EVENT_DEFINITIONS = {
    "Dot": {
        "runs": 0,
        "wicket": 0,
        "legal_ball": 1,
        "event_label": "Dot ball",
        "event_type": "dot",
        "is_boundary": False,
    },
    "1": {
        "runs": 1,
        "wicket": 0,
        "legal_ball": 1,
        "event_label": "Single",
        "event_type": "run",
        "is_boundary": False,
    },
    "2": {
        "runs": 2,
        "wicket": 0,
        "legal_ball": 1,
        "event_label": "Two runs",
        "event_type": "run",
        "is_boundary": False,
    },
    "3": {
        "runs": 3,
        "wicket": 0,
        "legal_ball": 1,
        "event_label": "Three runs",
        "event_type": "run",
        "is_boundary": False,
    },
    "4": {
        "runs": 4,
        "wicket": 0,
        "legal_ball": 1,
        "event_label": "FOUR",
        "event_type": "boundary",
        "is_boundary": True,
    },
    "6": {
        "runs": 6,
        "wicket": 0,
        "legal_ball": 1,
        "event_label": "SIX",
        "event_type": "boundary",
        "is_boundary": True,
    },
    "Wd": {
        "runs": 1,
        "wicket": 0,
        "legal_ball": 0,
        "event_label": "Wide",
        "event_type": "extra",
        "is_boundary": False,
    },
    "Nb": {
        "runs": 1,
        "wicket": 0,
        "legal_ball": 0,
        "event_label": "No ball",
        "event_type": "extra",
        "is_boundary": False,
    },
    "W": {
        "runs": 0,
        "wicket": 1,
        "legal_ball": 1,
        "event_label": "WICKET",
        "event_type": "wicket",
        "is_boundary": False,
    },
}


def default_live_config():

    return {
        "batting_team": teams[0],
        "bowling_team": teams[1],
        "city": cities[0],
        "target": 180,
        "initial_score": 0,
        "initial_wickets": 0,
        "initial_balls_completed": 0,
    }


if "live_config" not in st.session_state:

    st.session_state.live_config = default_live_config()

if "live_events" not in st.session_state:

    st.session_state.live_events = []


def build_match_state(config, events):

    score = int(config["initial_score"])
    wickets = int(config["initial_wickets"])
    balls_completed = int(config["initial_balls_completed"])

    for event in events:

        score += int(event["runs"])
        wickets += int(event["wicket"])
        balls_completed += int(event["legal_ball"])

    return {
        "score": score,
        "wickets": min(wickets, 10),
        "balls_completed": min(balls_completed, 120),
    }


def evaluate_snapshot(
    config,
    events,
    score,
    wickets,
    balls_completed,
    event_label,
    event_type,
):

    overs = balls_to_overs_float(balls_completed)
    context = calculate_match_context(
        target=config["target"],
        score=score,
        overs=overs,
        wickets=wickets,
        recent_events=events,
    )
    input_df = build_prediction_input(
        batting_team=config["batting_team"],
        bowling_team=config["bowling_team"],
        city=config["city"],
        context=context,
    )
    loss_prob, win_prob = predict_win_probabilities(
        model=model,
        input_df=input_df,
        context=context,
    )

    return {
        "score": score,
        "wickets": wickets,
        "balls_completed": balls_completed,
        "overs": overs,
        "over_ball": format_balls_as_overs(balls_completed),
        "scoreline": f"{score}/{wickets}",
        "runs_remaining": int(context["runs_remaining"]),
        "balls_remaining": int(context["balls_remaining"]),
        "current_run_rate": round(context["current_run_rate"], 2),
        "required_run_rate": round(context["required_run_rate"], 2),
        "wickets_remaining": int(context["wickets_remaining"]),
        "phase": context["phase"],
        "last_30_runs": context["last_30_runs"],
        "last_30_wickets": context["last_30_wickets"],
        "last_30_dot_balls": int(context["last_30_dot_balls"]),
        "last_30_boundaries": int(context["last_30_boundaries"]),
        "win_probability": round(win_prob * 100, 2),
        "loss_probability": round(loss_prob * 100, 2),
        "event_label": event_label,
        "event_type": event_type,
        "context": context,
        "input_df": input_df,
    }


def build_live_history(config, events):

    history_rows = []
    base_state = build_match_state(config, [])

    history_rows.append(
        evaluate_snapshot(
            config=config,
            events=[],
            score=base_state["score"],
            wickets=base_state["wickets"],
            balls_completed=base_state["balls_completed"],
            event_label="Starting point",
            event_type="start",
        )
    )

    score = base_state["score"]
    wickets = base_state["wickets"]
    balls_completed = base_state["balls_completed"]

    for index, event in enumerate(events):

        score += int(event["runs"])
        wickets += int(event["wicket"])
        balls_completed += int(event["legal_ball"])

        history_rows.append(
            evaluate_snapshot(
                config=config,
                events=events[: index + 1],
                score=score,
                wickets=min(wickets, 10),
                balls_completed=min(balls_completed, 120),
                event_label=event["event_label"],
                event_type=event["event_type"],
            )
        )

    return pd.DataFrame(history_rows)


def is_match_complete(snapshot_row):

    return bool(
        snapshot_row["runs_remaining"] <= 0 or
        snapshot_row["balls_remaining"] <= 0 or
        snapshot_row["wickets_remaining"] <= 0
    )


def push_event(event_key):

    st.session_state.live_events.append(
        EVENT_DEFINITIONS[event_key].copy()
    )


def live_message(snapshot_row, batting_team, bowling_team):

    if snapshot_row["runs_remaining"] <= 0:

        return (
            f"{batting_team} have chased it down. "
            f"Target cleared with {snapshot_row['wickets_remaining']} wickets left."
        )

    if snapshot_row["balls_remaining"] == 0:

        return (
            f"{bowling_team} close the innings out. "
            f"{snapshot_row['runs_remaining']} runs were still needed."
        )

    if snapshot_row["wickets_remaining"] == 0:

        return (
            f"{bowling_team} bowl them out. "
            f"{snapshot_row['runs_remaining']} runs remained in the chase."
        )

    if snapshot_row["event_type"] == "wicket":

        return (
            f"Wicket changes the equation at {snapshot_row['over_ball']}. "
            f"{batting_team} are now {snapshot_row['scoreline']} chasing {snapshot_row['runs_remaining']}."
        )

    if snapshot_row["event_type"] == "boundary":

        return (
            f"Boundary pressure release at {snapshot_row['over_ball']}. "
            f"{batting_team} move to {snapshot_row['scoreline']}."
        )

    return (
        f"{batting_team} are {snapshot_row['scoreline']} after {snapshot_row['over_ball']} overs, "
        f"needing {snapshot_row['runs_remaining']} from {snapshot_row['balls_remaining']} balls."
    )


with st.sidebar:

    st.markdown("### Live Chase Setup")

    with st.form("live-match-setup"):

        batting_team = st.selectbox(
            "Batting Team",
            teams,
            index=teams.index(st.session_state.live_config["batting_team"]),
        )

        available_bowling_teams = [
            team
            for team in teams
            if team != batting_team
        ]

        saved_bowling_team = st.session_state.live_config["bowling_team"]
        bowling_index = 0

        if saved_bowling_team in available_bowling_teams:

            bowling_index = available_bowling_teams.index(saved_bowling_team)

        bowling_team = st.selectbox(
            "Bowling Team",
            available_bowling_teams,
            index=bowling_index,
        )

        city = st.selectbox(
            "City",
            cities,
            index=cities.index(st.session_state.live_config["city"]),
        )

        target = st.number_input(
            "Target",
            min_value=1,
            value=int(st.session_state.live_config["target"]),
        )

        initial_score = st.number_input(
            "Starting Score",
            min_value=0,
            value=int(st.session_state.live_config["initial_score"]),
        )

        starting_overs = st.selectbox(
            "Starting Overs",
            [f"{over}.{ball}" for over in range(20) for ball in range(6)] + ["20.0"],
            index=min(
                int(st.session_state.live_config["initial_balls_completed"]),
                120,
            ),
        )

        initial_wickets = st.number_input(
            "Starting Wickets",
            min_value=0,
            max_value=10,
            value=int(st.session_state.live_config["initial_wickets"]),
        )

        submitted = st.form_submit_button("Start New Chase")

    if submitted:

        over_part, ball_part = starting_overs.split(".")

        st.session_state.live_config = {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "city": city,
            "target": int(target),
            "initial_score": int(initial_score),
            "initial_wickets": int(initial_wickets),
            "initial_balls_completed": int(
                (int(over_part) * 6) + int(ball_part)
            ),
        }
        st.session_state.live_events = []
        st.rerun()

    st.caption(
        "The live page is realtime inside the app. It updates on every ball event you log."
    )

render_page_header(
    title="Realtime Chase Predictor",
    subtitle=(
        "Run the innings ball by ball, update score pressure instantly, and watch win probability "
        "swing like a broadcast match center."
    ),
    kicker="Live Match Center",
)

live_config = st.session_state.live_config
live_history_df = build_live_history(
    config=live_config,
    events=st.session_state.live_events,
)
current_snapshot = live_history_df.iloc[-1]
match_complete = is_match_complete(current_snapshot)

branding_columns = st.columns([1, 0.2, 1, 1])

with branding_columns[0]:

    render_team_pill(live_config["batting_team"])

with branding_columns[1]:

    st.markdown("## vs")

with branding_columns[2]:

    render_team_pill(live_config["bowling_team"])

with branding_columns[3]:

    render_metric_card(
        "Target",
        live_config["target"],
        f"{current_snapshot['runs_remaining']} still needed",
    )

score_columns = st.columns(5)

with score_columns[0]:

    render_metric_card(
        "Current Score",
        current_snapshot["scoreline"],
        f"After {current_snapshot['over_ball']} overs",
    )

with score_columns[1]:

    render_metric_card(
        live_config["batting_team"],
        f"{current_snapshot['win_probability']:.1f}%",
        "Projected chase success",
    )

with score_columns[2]:

    render_metric_card(
        live_config["bowling_team"],
        f"{current_snapshot['loss_probability']:.1f}%",
        "Projected defense success",
    )

with score_columns[3]:

    render_metric_card(
        "Run Rates",
        f"{current_snapshot['current_run_rate']:.2f}",
        f"Required {current_snapshot['required_run_rate']:.2f}",
    )

with score_columns[4]:

    render_metric_card(
        "Phase",
        current_snapshot["phase"],
        f"{current_snapshot['balls_remaining']} balls left",
    )

st.info(
    live_message(
        snapshot_row=current_snapshot,
        batting_team=live_config["batting_team"],
        bowling_team=live_config["bowling_team"],
    )
)

control_rows = [
    ["Dot", "1", "2", "4", "W"],
    ["3", "6", "Wd", "Nb", "Undo"],
]

for row_index, button_row in enumerate(control_rows):

    action_columns = st.columns(5)

    for column, button_label in zip(action_columns, button_row):

        with column:

            if button_label == "Undo":

                if st.button(
                    "Undo Last Ball",
                    use_container_width=True,
                    disabled=not st.session_state.live_events,
                    key=f"undo-{row_index}",
                ):

                    st.session_state.live_events.pop()
                    st.rerun()

            else:

                if st.button(
                    button_label,
                    use_container_width=True,
                    disabled=match_complete,
                    key=f"event-{button_label}-{row_index}",
                ):

                    push_event(button_label)
                    st.rerun()

if st.button(
    "Reset Logged Events",
    disabled=not st.session_state.live_events,
):

    st.session_state.live_events = []
    st.rerun()

context = current_snapshot["context"]
momentum_index = compute_momentum_index(
    context["current_run_rate"],
    context["required_run_rate"],
    context["wickets_remaining"],
    context["last_30_runs"],
    context["last_30_dot_balls"],
    context["last_30_boundaries"],
)

pressure_index = round(
    context["required_run_rate"] - context["current_run_rate"],
    2,
)

attack_index = round(
    (context["last_30_runs"] / 5) +
    ((context["last_30_boundaries"] * 2) / 10) -
    (context["last_30_dot_balls"] * 2),
    1,
)

momentum_columns = st.columns(4)

momentum_columns[0].metric(
    "Momentum Index",
    f"{momentum_index:+.1f}",
    get_momentum_label(momentum_index),
)
momentum_columns[1].metric(
    "Run Rate Pressure",
    f"{pressure_index:+.2f}",
    "Required RR - Current RR",
    delta_color="inverse",
)
momentum_columns[2].metric(
    "Attack Index",
    f"{attack_index:.1f}",
    f"{context['last_30_boundaries']} boundaries in the last burst",
)
momentum_columns[3].metric(
    "Dot Ball Pressure",
    int(context["last_30_dot_balls"]),
    f"{context['wickets_remaining']} wickets in hand",
    delta_color="inverse",
)

chart_columns = st.columns([1.7, 1])

with chart_columns[0]:

    st.plotly_chart(
        create_live_match_figure(
            history_df=live_history_df,
            batting_color=get_team_color(live_config["batting_team"]),
            bowling_color=get_team_color(
                live_config["bowling_team"],
                "#dc2626",
            ),
        ),
        use_container_width=True,
    )

with chart_columns[1]:

    st.markdown("### Momentum Read")
    st.progress((momentum_index + 100) / 200)
    st.write(f"**{get_momentum_label(momentum_index)}**")

    for driver in build_momentum_drivers(
        batting_team=live_config["batting_team"],
        bowling_team=live_config["bowling_team"],
        pressure_index=pressure_index,
        current_run_rate=context["current_run_rate"],
        required_run_rate=context["required_run_rate"],
        last_30_boundaries=context["last_30_boundaries"],
        last_30_dot_balls=context["last_30_dot_balls"],
        wickets_remaining=context["wickets_remaining"],
    ):

        st.write(f"- {driver}")

    st.markdown("### AI Notes")

    for insight in build_ai_insights(
        current_run_rate=context["current_run_rate"],
        required_run_rate=context["required_run_rate"],
        wickets_remaining=context["wickets_remaining"],
        balls_remaining=context["balls_remaining"],
        last_30_dot_balls=context["last_30_dot_balls"],
        last_30_boundaries=context["last_30_boundaries"],
    ):

        st.write(f"- {insight}")

with st.expander("Why the model says this right now"):

    transformed_input = preprocessor.transform(
        current_snapshot["input_df"]
    )

    if hasattr(transformed_input, "toarray"):

        transformed_input = transformed_input.toarray()

    explainer = shap.TreeExplainer(xgb_model)

    st.pyplot(
        create_shap_waterfall_figure(
            explainer=explainer,
            transformed_input=transformed_input,
            feature_names=preprocessor.get_feature_names_out(),
        )
    )

st.markdown("### Logged Ball Events")

if len(live_history_df) > 1:

    event_table_df = live_history_df.iloc[1:].copy()
    event_table_df["Batting Win %"] = event_table_df["win_probability"]

    st.dataframe(
        event_table_df.rename(columns={
            "over_ball": "Over",
            "event_label": "Event",
            "scoreline": "Score",
            "runs_remaining": "Runs Left",
            "balls_remaining": "Balls Left",
            "current_run_rate": "CRR",
            "required_run_rate": "RRR",
        })[
            [
                "Over",
                "Event",
                "Score",
                "Runs Left",
                "Balls Left",
                "CRR",
                "RRR",
                "Batting Win %",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

else:

    st.caption("No balls logged yet. Start the chase with the event buttons above.")
