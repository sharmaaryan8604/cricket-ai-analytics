import streamlit as st

from utils.charts import (
    create_caps_trend_figure,
    create_leaderboard_bar_figure,
)
from utils.dashboard import (
    get_batting_leaders,
    get_bowling_leaders,
    get_season_caps,
    get_season_overview,
    latest_season,
)
from utils.ui import (
    apply_app_styles,
    render_metric_card,
    render_page_header,
)


st.set_page_config(
    page_title="Season Leaders",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styles()

batting_leaders_df = get_batting_leaders()
bowling_leaders_df = get_bowling_leaders()
season_caps_df = get_season_caps()
season_overview_df = get_season_overview()

available_seasons = sorted(
    season_caps_df["season"].dropna().astype(int).unique().tolist(),
    reverse=True,
)
default_season = latest_season()

render_page_header(
    title="Season-Wise Run and Wicket Leaders",
    subtitle=(
        "Explore every season like an IPL dashboard: top run scorer, top wicket taker, "
        "and the full leaderboard behind each cap race."
    ),
    kicker="Orange Cap + Purple Cap",
)

selected_season = st.selectbox(
    "Choose a season",
    available_seasons,
    index=available_seasons.index(default_season),
)

season_caps_row = season_caps_df[
    season_caps_df["season"] == selected_season
].iloc[0]

season_summary_row = season_overview_df[
    season_overview_df["season"] == selected_season
].iloc[0]

top_batters_df = (
    batting_leaders_df[
        batting_leaders_df["season"] == selected_season
    ]
    .head(10)
    .copy()
)
top_bowlers_df = (
    bowling_leaders_df[
        bowling_leaders_df["season"] == selected_season
    ]
    .head(10)
    .copy()
)

summary_columns = st.columns(4)

with summary_columns[0]:

    render_metric_card(
        "Top Run Scorer",
        season_caps_row["orange_cap"],
        f"{int(season_caps_row['orange_runs'])} runs for {season_caps_row['orange_team']}",
    )

with summary_columns[1]:

    render_metric_card(
        "Top Wicket Taker",
        season_caps_row["purple_cap"],
        f"{int(season_caps_row['purple_wickets'])} wickets for {season_caps_row['purple_team']}",
    )

with summary_columns[2]:

    render_metric_card(
        "Season Matches",
        int(season_summary_row["matches"]),
        f"{int(season_summary_row['venues'])} venues in use",
    )

with summary_columns[3]:

    render_metric_card(
        "Runs + Wickets",
        f"{int(season_summary_row['total_runs'])}",
        f"{int(season_summary_row['total_wickets'])} wickets recorded",
    )

trend_column, latest_caps_column = st.columns([1.45, 1])

with trend_column:

    st.plotly_chart(
        create_caps_trend_figure(season_caps_df),
        use_container_width=True,
    )

with latest_caps_column:

    season_caps_table_df = season_caps_df.rename(columns={
        "season": "Season",
        "orange_cap": "Top Run Scorer",
        "orange_runs": "Runs",
        "purple_cap": "Top Wicket Taker",
        "purple_wickets": "Wickets",
    })

    st.dataframe(
        season_caps_table_df[
            [
                "Season",
                "Top Run Scorer",
                "Runs",
                "Top Wicket Taker",
                "Wickets",
            ]
        ].sort_values("Season", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

leaderboard_columns = st.columns(2)

with leaderboard_columns[0]:

    st.plotly_chart(
        create_leaderboard_bar_figure(
            leaderboard_df=top_batters_df,
            player_column="player",
            value_column="runs",
            title=f"{selected_season} Orange Cap Race",
            color="#f59e0b",
        ),
        use_container_width=True,
    )

    st.dataframe(
        top_batters_df.rename(columns={
            "rank": "Rank",
            "player": "Player",
            "team": "Team",
            "runs": "Runs",
            "balls": "Balls",
            "strike_rate": "Strike Rate",
            "fours": "4s",
            "sixes": "6s",
            "matches": "Matches",
        })[
            [
                "Rank",
                "Player",
                "Team",
                "Runs",
                "Balls",
                "Strike Rate",
                "4s",
                "6s",
                "Matches",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with leaderboard_columns[1]:

    st.plotly_chart(
        create_leaderboard_bar_figure(
            leaderboard_df=top_bowlers_df,
            player_column="player",
            value_column="wickets",
            title=f"{selected_season} Purple Cap Race",
            color="#8b5cf6",
        ),
        use_container_width=True,
    )

    st.dataframe(
        top_bowlers_df.rename(columns={
            "rank": "Rank",
            "player": "Player",
            "team": "Team",
            "wickets": "Wickets",
            "overs": "Overs",
            "economy": "Economy",
            "strike_rate": "Strike Rate",
            "matches": "Matches",
        })[
            [
                "Rank",
                "Player",
                "Team",
                "Wickets",
                "Overs",
                "Economy",
                "Strike Rate",
                "Matches",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
