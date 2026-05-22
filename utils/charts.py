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
