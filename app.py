import time

import shap
import streamlit as st

from utils.charts import (
    create_live_replay_figure,
    create_momentum_figure,
    create_shap_waterfall_figure,
)
from utils.commentary import (
    build_ai_insights,
    build_live_commentary,
    build_momentum_drivers,
)
from utils.prediction import (
    CITIES,
    TEAM_LOGOS,
    TEAMS,
    build_prediction_input,
    build_valid_overs,
    calculate_match_context,
    compute_momentum_index,
    get_momentum_label,
    load_ball_data,
    load_model,
)
from utils.replay import (
    filter_match_replay,
    generate_replay_data,
    prepare_match_options,
)


st.set_page_config(
    page_title="Explainable IPL Win Predictor",
    layout="wide",
)


@st.cache_resource
def get_model():

    return load_model()


@st.cache_data
def get_ball_df():

    return load_ball_data()


model = get_model()
ball_df = get_ball_df()

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["classifier"]


st.title("Explainable IPL Win Probability Engine")
st.write("Real-time IPL match prediction using XGBoost + SHAP")

col1, col2 = st.columns(2)

with col1:

    batting_team = st.selectbox(
        "Batting Team",
        TEAMS,
    )

    bowling_team = st.selectbox(
        "Bowling Team",
        TEAMS,
    )

    city = st.selectbox(
        "Match City",
        CITIES,
    )

with col2:

    target = st.number_input(
        "Target Score",
        min_value=1,
        value=180,
    )

    score = st.number_input(
        "Current Score",
        min_value=0,
        value=100,
    )

    overs = st.selectbox(
        "Overs Completed",
        build_valid_overs(),
        index=60,
    )

    overs = float(overs)

    wickets = st.number_input(
        "Wickets Fallen",
        min_value=0,
        max_value=10,
        value=3,
    )

logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])

with logo_col1:

    if batting_team in TEAM_LOGOS:

        st.image(
            str(TEAM_LOGOS[batting_team]),
            width=120,
        )

with logo_col2:

    st.markdown(
        "<h1 style='text-align:center;'>VS</h1>",
        unsafe_allow_html=True,
    )

with logo_col3:

    if bowling_team in TEAM_LOGOS:

        st.image(
            str(TEAM_LOGOS[bowling_team]),
            width=120,
        )

if batting_team == bowling_team:

    st.error("Batting and Bowling teams cannot be same.")
    st.stop()

if st.button("Predict Win Probability"):

    context = calculate_match_context(
        target=target,
        score=score,
        overs=overs,
        wickets=wickets,
    )

    input_df = build_prediction_input(
        batting_team=batting_team,
        bowling_team=bowling_team,
        city=city,
        context=context,
    )

    result = model.predict_proba(input_df)
    loss_prob = result[0][0]
    win_prob = result[0][1]

    st.header(f"{batting_team} Win Probability")
    st.progress(float(win_prob))
    st.success(f"{round(win_prob * 100)}%")

    st.header(f"{bowling_team} Win Probability")
    st.progress(float(loss_prob))
    st.error(f"{round(loss_prob * 100)}%")

    st.header("Momentum Analytics")

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

    momentum_cols = st.columns(4)

    momentum_cols[0].metric(
        "Momentum Index",
        f"{momentum_index:+.1f}",
        get_momentum_label(momentum_index),
    )

    momentum_cols[1].metric(
        "Run Rate Pressure",
        f"{pressure_index:+.2f}",
        "Required RR - Current RR",
        delta_color="inverse",
    )

    momentum_cols[2].metric(
        "Attack Index",
        f"{attack_index:.1f}",
        f"{context['last_30_boundaries']} boundaries",
    )

    momentum_cols[3].metric(
        "Dot Ball Pressure",
        int(context["last_30_dot_balls"]),
        f"{context['wickets_remaining']} wickets left",
        delta_color="inverse",
    )

    st.progress((momentum_index + 100) / 200)

    st.write(
        f"Momentum verdict: **{get_momentum_label(momentum_index)}**"
    )

    for driver in build_momentum_drivers(
        batting_team=batting_team,
        bowling_team=bowling_team,
        pressure_index=pressure_index,
        current_run_rate=context["current_run_rate"],
        required_run_rate=context["required_run_rate"],
        last_30_boundaries=context["last_30_boundaries"],
        last_30_dot_balls=context["last_30_dot_balls"],
        wickets_remaining=context["wickets_remaining"],
    ):

        st.write(f"- {driver}")

    st.header("Why Did The Model Predict This?")

    transformed_input = preprocessor.transform(input_df).toarray()
    explainer = shap.TreeExplainer(xgb_model)

    shap_fig = create_shap_waterfall_figure(
        explainer=explainer,
        transformed_input=transformed_input,
        feature_names=preprocessor.get_feature_names_out(),
    )

    st.pyplot(shap_fig)

    st.header("AI Match Insights")

    for insight in build_ai_insights(
        current_run_rate=context["current_run_rate"],
        required_run_rate=context["required_run_rate"],
        wickets_remaining=context["wickets_remaining"],
        balls_remaining=context["balls_remaining"],
        last_30_dot_balls=context["last_30_dot_balls"],
        last_30_boundaries=context["last_30_boundaries"],
    ):

        st.write(f"- {insight}")

st.markdown("---")
st.header("Historical Match Replay")

match_options = prepare_match_options(ball_df)
match_labels = dict(
    zip(
        match_options["match_id"],
        match_options["match_label"],
    )
)

selected_match = st.selectbox(
    "Select Historical Match",
    match_options["match_id"],
    format_func=lambda match_id: match_labels.get(
        match_id,
        str(match_id),
    ),
)

match_df = filter_match_replay(
    ball_df=ball_df,
    selected_match=selected_match,
)

replay_data = generate_replay_data(
    match_df=match_df,
    model=model,
)

if replay_data:

    st.subheader("Momentum Analytics")

    replay_metrics = st.columns(3)

    replay_metrics[0].metric(
        "Net Momentum Swing",
        replay_data["net_momentum_display"],
        "Total probability movement",
    )

    replay_metrics[1].metric(
        "Biggest Ball Swing",
        replay_data["biggest_swing_display"],
        replay_data["biggest_swing_row"]["over_ball"],
    )

    replay_metrics[2].metric(
        "Pressure Split",
        f"{replay_data['batting_push_balls']}-{replay_data['bowling_push_balls']}",
        "Batting pushes vs bowling pushes",
    )

    st.write(
        f"Biggest momentum moment: **{replay_data['biggest_swing_row']['over_ball']}** when "
        f"**{replay_data['swing_team']}** grabbed "
        f"**{abs(replay_data['biggest_swing_row']['momentum_shift']):.2f}** "
        "win-probability points."
    )

    momentum_fig = create_momentum_figure(
        replay_df=replay_data["replay_df"],
        batting_color=replay_data["replay_batting_color"],
        bowling_color=replay_data["replay_bowling_color"],
    )

    st.plotly_chart(
        momentum_fig,
        use_container_width=True,
    )

else:

    st.warning(
        "Momentum analytics are unavailable for this match because no replay data could be generated."
    )

st.subheader("Animated Ball-by-Ball Replay")

graph_placeholder = st.empty()
commentary_placeholder = st.empty()
current_prob_text = st.empty()

speed = st.slider(
    "Replay Speed",
    0.01,
    1.0,
    0.1,
)

if replay_data:

    live_x = []
    live_y = []

    for index, probability in enumerate(replay_data["probabilities"]):

        live_x.append(replay_data["overs_progress"][index])
        live_y.append(probability)

        current_events_x = []
        current_events_y = []
        current_events_text = []
        current_event_colors = []

        for event_index, event_ball in enumerate(replay_data["event_x"]):

            if event_ball in live_x:

                current_events_x.append(event_ball)
                current_events_y.append(replay_data["event_y"][event_index])
                current_events_text.append(replay_data["event_text"][event_index])

                current_event_colors.append(
                    replay_data["replay_bowling_color"]
                    if "WICKET" in replay_data["event_text"][event_index]
                    else replay_data["replay_batting_color"]
                )

        live_fig = create_live_replay_figure(
            live_x=live_x,
            live_y=live_y,
            current_events_x=current_events_x,
            current_events_y=current_events_y,
            current_events_text=current_events_text,
            current_event_colors=current_event_colors,
            batting_color=replay_data["replay_batting_color"],
            bowling_color=replay_data["replay_bowling_color"],
        )

        graph_placeholder.plotly_chart(
            live_fig,
            use_container_width=True,
        )

        current_prob_text.metric(
            "Current Win Probability",
            f"{round(probability, 2)}%",
        )

        current_ball = live_x[-1]
        match_row = match_df.iloc[index]

        commentary_placeholder.info(
            build_live_commentary(
                match_row=match_row,
                current_ball=current_ball,
            )
        )

        time.sleep(speed)

st.markdown("---")
st.caption(
    "Built using XGBoost, SHAP, Streamlit and IPL Ball-by-Ball Data"
)
