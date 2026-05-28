import streamlit as st

from utils.prediction import (
    get_team_color,
    get_team_logo_path,
    get_team_short_name,
)


APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(245, 158, 11, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.16), transparent 24%),
            linear-gradient(180deg, #08111f 0%, #101a2e 52%, #0b1322 100%);
        color: #f8fafc;
        font-family: "Aptos", "Trebuchet MS", "Segoe UI", sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 19, 34, 0.98), rgba(16, 26, 46, 0.98));
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.88));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 24px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 18px 40px rgba(8, 15, 28, 0.32);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        color: #f59e0b;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        font-size: 2.35rem;
        line-height: 1.08;
        font-weight: 800;
        margin: 0;
        color: #f8fafc;
    }

    .hero-subtitle {
        margin-top: 0.8rem;
        font-size: 1.02rem;
        line-height: 1.6;
        color: #cbd5e1;
        max-width: 960px;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        min-height: 128px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 700;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 0.55rem;
    }

    .metric-helper {
        color: #cbd5e1;
        margin-top: 0.55rem;
        font-size: 0.94rem;
    }

    .section-copy {
        color: #cbd5e1;
        margin-top: -0.25rem;
        margin-bottom: 0.8rem;
    }

    .team-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
        border-radius: 999px;
        padding: 0.45rem 0.85rem;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.14);
        margin-bottom: 0.5rem;
    }

    .team-mark {
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 800;
        color: #ffffff;
    }

    .team-name {
        color: #f8fafc;
        font-weight: 700;
    }
</style>
"""


def apply_app_styles():

    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_page_header(title, subtitle, kicker="IPL Dashboard"):

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, helper):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_team_pill(team_name):

    team_color = get_team_color(team_name)
    team_short_name = get_team_short_name(team_name)
    team_logo = get_team_logo_path(team_name)

    if team_logo:

        st.image(str(team_logo), width=64)

    st.markdown(
        f"""
        <div class="team-pill">
            <div class="team-mark" style="background:{team_color};">
                {team_short_name}
            </div>
            <div class="team-name">{team_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
