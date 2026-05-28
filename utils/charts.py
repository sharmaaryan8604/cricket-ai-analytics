import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap


def hex_to_rgba(hex_color, alpha):

    hex_color = hex_color.lstrip("#")

    if len(hex_color) != 6:

        return f"rgba(0, 191, 255, {alpha})"

    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)

    return f"rgba({red}, {green}, {blue}, {alpha})"


def create_shap_waterfall_figure(
    explainer,
    transformed_input,
    feature_names,
):

    shap_values = explainer.shap_values(transformed_input)

    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=transformed_input[0],
        feature_names=feature_names,
    )

    fig, _ = plt.subplots(figsize=(12, 6))

    shap.plots.waterfall(
        shap_explanation,
        max_display=10,
        show=False,
    )

    return fig


def create_momentum_figure(
    replay_df,
    batting_color,
    bowling_color,
):

    momentum_fig = go.Figure()

    momentum_fig.add_trace(
        go.Bar(
            x=replay_df["over_ball"],
            y=replay_df["momentum_shift"],
            marker_color=replay_df["momentum_color"],
            marker_line_color=replay_df["momentum_color"],
            marker_line_width=0.4,
            name="Ball-by-ball swing",
        )
    )

    momentum_fig.add_trace(
        go.Scatter(
            x=replay_df["over_ball"],
            y=replay_df["rolling_momentum"],
            mode="lines",
            name="Rolling 6-ball swing",
            line=dict(
                color=batting_color,
                width=3,
            ),
        )
    )

    momentum_fig.update_layout(
        title="Momentum Shift Timeline",
        xaxis_title="Overs",
        yaxis_title="Momentum Shift (pp)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(12, 18, 28, 0.92)",
        font=dict(color="#f8fafc"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor=hex_to_rgba(
                bowling_color,
                0.45,
            ),
        ),
        height=420,
    )

    return momentum_fig


def create_live_replay_figure(
    live_x,
    live_y,
    current_events_x,
    current_events_y,
    current_events_text,
    current_event_colors,
    batting_color,
    bowling_color,
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=live_x,
            y=live_y,
            mode="lines",
            name="Win Probability",
            line=dict(
                width=4,
                color=batting_color,
            ),
            fill="tozeroy",
            fillcolor=hex_to_rgba(
                batting_color,
                0.2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=current_events_x,
            y=current_events_y,
            mode="markers+text",
            text=current_events_text,
            textposition="top center",
            marker=dict(
                size=10,
                color=current_event_colors,
                line=dict(
                    color="#ffffff",
                    width=1,
                ),
            ),
            textfont=dict(color="#f8fafc"),
            name="Match Events",
        )
    )

    fig.update_layout(
        title="Animated Win Probability Replay",
        xaxis_title="Overs",
        yaxis_title="Win Probability %",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(12, 18, 28, 0.92)",
        font=dict(color="#f8fafc"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor=hex_to_rgba(
                bowling_color,
                0.45,
            ),
        ),
        height=650,
    )

    return fig


def create_leaderboard_bar_figure(
    leaderboard_df,
    player_column,
    value_column,
    title,
    color,
    suffix="",
):

    chart_df = leaderboard_df.iloc[::-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df[value_column],
            y=chart_df[player_column],
            orientation="h",
            marker=dict(
                color=color,
                line=dict(
                    color="#f8fafc",
                    width=0.8,
                ),
            ),
            text=[
                f"{value}{suffix}"
                for value in chart_df[value_column]
            ],
            textposition="outside",
            hovertemplate="%{y}: %{x}" + suffix + "<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(12, 18, 28, 0.92)",
        font=dict(color="#f8fafc"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zeroline=False,
            title="",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            title="",
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        height=420,
    )

    return fig


def create_caps_trend_figure(caps_df):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=caps_df["season"],
            y=caps_df["orange_runs"],
            mode="lines+markers",
            name="Orange Cap Runs",
            line=dict(
                color="#f59e0b",
                width=3,
            ),
            marker=dict(size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=caps_df["season"],
            y=caps_df["purple_wickets"],
            mode="lines+markers",
            name="Purple Cap Wickets",
            line=dict(
                color="#8b5cf6",
                width=3,
            ),
            marker=dict(size=8),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Season Leaders Trend",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(12, 18, 28, 0.92)",
        font=dict(color="#f8fafc"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zeroline=False,
            title="Season",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zeroline=False,
            title="Runs",
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            title="Wickets",
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=440,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    return fig


def create_live_match_figure(
    history_df,
    batting_color,
    bowling_color,
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["over_ball"],
            y=history_df["win_probability"],
            mode="lines+markers",
            name="Win Probability",
            line=dict(
                color=batting_color,
                width=4,
            ),
            marker=dict(size=8, color=batting_color),
            fill="tozeroy",
            fillcolor=hex_to_rgba(
                batting_color,
                0.18,
            ),
            hovertemplate=(
                "Over %{x}<br>"
                "Win probability %{y:.1f}%<br>"
                "Score %{customdata[0]}<extra></extra>"
            ),
            customdata=history_df[["scoreline"]],
        )
    )

    event_df = history_df[
        history_df["event_type"].isin(["boundary", "wicket"])
    ].copy()

    if not event_df.empty:

        fig.add_trace(
            go.Scatter(
                x=event_df["over_ball"],
                y=event_df["win_probability"],
                mode="markers+text",
                text=event_df["event_label"],
                textposition="top center",
                name="Key Moments",
                marker=dict(
                    size=12,
                    color=[
                        bowling_color
                        if event_type == "wicket"
                        else batting_color
                        for event_type in event_df["event_type"]
                    ],
                    line=dict(
                        color="#ffffff",
                        width=1,
                    ),
                ),
                textfont=dict(color="#f8fafc"),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Live Chase Probability Curve",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(12, 18, 28, 0.92)",
        font=dict(color="#f8fafc"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zeroline=False,
            title="Overs",
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor=hex_to_rgba(
                bowling_color,
                0.45,
            ),
            title="Win Probability %",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        height=500,
    )

    return fig
