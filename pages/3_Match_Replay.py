import time

import streamlit as st

from utils.charts import (
    create_live_replay_figure,
    create_momentum_figure,
)
from utils.commentary import build_live_commentary
from utils.dashboard import (
    get_ball_df,
    get_model,
)
from utils.replay import (
    filter_match_replay,
    generate_replay_data,
    prepare_match_options,
)
from utils.ui import (
    apply_app_styles,
    render_metric_card,
    render_page_header,
)


st.set_page_config(
    page_title="Historical Match Replay",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styles()

ball_df = get_ball_df()
model = get_model()
match_options_df = prepare_match_options(ball_df)

available_seasons = sorted(
    match_options_df["season"]
    .dropna()
    .astype(int)
    .unique()
    .tolist(),
    reverse=True,
)

render_page_header(
    title="Historical Match Replay",
    subtitle=(
        "Rewind second-innings chases, inspect momentum shifts, and replay the win-probability curve "
        "ball by ball."
    ),
    kicker="Replay Center",
)

season_filter = st.selectbox(
    "Filter by season",
    available_seasons,
)

season_match_options_df = match_options_df[
    match_options_df["season"] == season_filter
].copy()

match_labels = dict(
    zip(
        season_match_options_df["match_id"],
        season_match_options_df["match_label"],
    )
)

selected_match = st.selectbox(
    "Select match",
    season_match_options_df["match_id"],
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

if not replay_data:

    st.warning(
        "Replay data is unavailable for this match."
    )
    st.stop()

metric_columns = st.columns(4)

with metric_columns[0]:

    render_metric_card(
        "Net Momentum",
        replay_data["net_momentum_display"],
        "Total win-probability movement",
    )

with metric_columns[1]:

    render_metric_card(
        "Biggest Ball Swing",
        replay_data["biggest_swing_display"],
        replay_data["biggest_swing_row"]["over_ball"],
    )

with metric_columns[2]:

    render_metric_card(
        "Pressure Split",
        f"{replay_data['batting_push_balls']}-{replay_data['bowling_push_balls']}",
        "Batting pushes vs bowling pushes",
    )

with metric_columns[3]:

    render_metric_card(
        "Replay Balls",
        len(replay_data["probabilities"]),
        f"{replay_data['replay_batting_team']} chase",
    )

st.info(
    f"Biggest turning point: {replay_data['biggest_swing_row']['over_ball']} when "
    f"{replay_data['swing_team']} grabbed {abs(replay_data['biggest_swing_row']['momentum_shift']):.2f} "
    "win-probability points."
)

st.plotly_chart(
    create_momentum_figure(
        replay_df=replay_data["replay_df"],
        batting_color=replay_data["replay_batting_color"],
        bowling_color=replay_data["replay_bowling_color"],
    ),
    use_container_width=True,
)

st.markdown("### Animated Ball-by-Ball Replay")
speed = st.slider(
    "Replay speed",
    min_value=0.01,
    max_value=0.6,
    value=0.08,
)

graph_placeholder = st.empty()
commentary_placeholder = st.empty()
current_prob_placeholder = st.empty()

if st.button("Play Replay"):

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

        graph_placeholder.plotly_chart(
            create_live_replay_figure(
                live_x=live_x,
                live_y=live_y,
                current_events_x=current_events_x,
                current_events_y=current_events_y,
                current_events_text=current_events_text,
                current_event_colors=current_event_colors,
                batting_color=replay_data["replay_batting_color"],
                bowling_color=replay_data["replay_bowling_color"],
            ),
            use_container_width=True,
        )

        current_prob_placeholder.metric(
            "Current Win Probability",
            f"{round(probability, 2)}%",
        )

        commentary_placeholder.info(
            build_live_commentary(
                match_row=match_df.iloc[index],
                current_ball=live_x[-1],
            )
        )

        time.sleep(speed)

else:

    graph_placeholder.plotly_chart(
        create_live_replay_figure(
            live_x=replay_data["overs_progress"][:1],
            live_y=replay_data["probabilities"][:1],
            current_events_x=[],
            current_events_y=[],
            current_events_text=[],
            current_event_colors=[],
            batting_color=replay_data["replay_batting_color"],
            bowling_color=replay_data["replay_bowling_color"],
        ),
        use_container_width=True,
    )
