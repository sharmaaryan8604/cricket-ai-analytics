# IPL Analytics Dashboard

A multi-page Streamlit app for IPL-style cricket analytics with:

- realtime chase prediction
- season-wise highest run scorer tracking
- season-wise highest wicket taker tracking
- historical match replay with momentum swings
- SHAP-based model explainability

## Pages

### Home
- season overview
- latest Orange Cap and Purple Cap winners
- top performers for the latest season

### Live Predictor
- ball-by-ball chase controls
- instant win probability updates
- momentum and pressure metrics
- SHAP explanation for the current match state

### Season Leaders
- season selector
- top run scorer for every season
- top wicket taker for every season
- full top-10 batting and bowling leaderboards

### Match Replay
- historical second-innings replay
- momentum timeline
- animated win-probability playback

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Matplotlib

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

This repo is ready for Streamlit Community Cloud deployment from GitHub.

1. Push this branch to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set the main file path to `app.py`.
4. Let Streamlit install from `requirements.txt` and `runtime.txt`.

Included deployment files:

- `.streamlit/config.toml`
- `runtime.txt`

## Project Structure

```text
cricket_ai/
|-- app.py
|-- pages/
|   |-- 1_Live_Predictor.py
|   |-- 2_Season_Leaders.py
|   |-- 3_Match_Replay.py
|-- data/
|   |-- ipl_ball_by_ball.csv
|-- models/
|   |-- xgboost_win_predictor.pkl
|-- utils/
|   |-- charts.py
|   |-- commentary.py
|   |-- dashboard.py
|   |-- prediction.py
|   |-- replay.py
|   |-- ui.py
|-- images/
|-- requirements.txt
```

## Notes

- The live page is realtime inside the app and updates whenever you log a ball event.
- Season values are derived from match dates in the dataset.
- Bowling leaderboards use wicket-credit deltas from the cumulative bowler wicket field.
