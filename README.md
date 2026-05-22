# 🏏 Explainable IPL Win Probability Engine

An AI-powered cricket analytics platform that predicts IPL match outcomes using Machine Learning, XGBoost, SHAP explainability, momentum analysis, and interactive visualizations.

The project combines sports analytics, explainable AI, and real-time match simulation to create a modern cricket intelligence dashboard inspired by professional broadcast analytics systems.

---

# 🚀 Features

## 📈 Real-Time Win Probability Prediction
Predicts the probability of a team winning during a run chase using:
- current score
- wickets remaining
- overs completed
- current run rate
- required run rate

Built using an XGBoost classification model trained on historical IPL ball-by-ball data.

---

## 🎥 Historical Match Replay
Replay historical IPL matches ball-by-ball with:
- changing win probability
- live momentum shifts
- event-based commentary
- turning point detection

---

## 📊 Momentum Analytics
Analyze:
- biggest momentum swings
- pressure phases
- match turning points
- batting vs bowling dominance

---

## 🧠 Explainable AI with SHAP
Visualize why the model predicts a team will win or lose using SHAP feature explanations.

The app highlights:
- feature importance
- match pressure indicators
- impact of wickets and required run rate

---

## 🎨 IPL Broadcast-Style UI
Includes:
- IPL team logos
- team color themes
- dark analytics dashboard
- animated replay visuals
- interactive Plotly charts

---

## ⚡ Animated Ball-by-Ball Replay
Interactive replay engine with:
- probability progression
- event markers
- fours, sixes, wickets
- live-style commentary simulation

---

# 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core Development |
| Streamlit | Frontend Web App |
| XGBoost | ML Prediction Model |
| SHAP | Explainable AI |
| Plotly | Interactive Visualizations |
| Pandas | Data Processing |
| Scikit-learn | ML Pipeline |

---

# 📂 Project Structure

```bash
cricket_ai/

│
├── app.py
│
├── api/
│
├── data/
│   ├── ipl_ball_by_ball.csv
│   ├── ipl_win_prediction_dataset.csv
│
├── images/
│   └── logos/
│
├── models/
│   ├── xgboost_win_predictor.pkl
│   ├── random_forest_win_predictor.pkl
│
├── notebooks/
│   └── datacreation.py
│
├── utils/
│   ├── charts.py
│   ├── commentary.py
│   ├── prediction.py
│   ├── replay.py
│
├── requirements.txt
│
└── README.md