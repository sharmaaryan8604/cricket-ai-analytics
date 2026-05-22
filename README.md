# 🏏 Explainable IPL Win Probability Engine

An AI-powered cricket analytics platform that predicts IPL match outcomes using Machine Learning, XGBoost, SHAP explainability, momentum analysis, and interactive visualizations.

This project combines:
- Sports Analytics
- Explainable AI
- Interactive Data Visualization
- Real-Time Match Intelligence

to create a modern cricket analytics dashboard inspired by professional broadcast systems.

---
# 🌐 Live Demo

https://cricket-ai-analytics.streamlit.app/

# 🚀 Features

## 📈 Win Probability Prediction
Predicts the probability of a team winning during a run chase using:
- current score
- wickets remaining
- overs completed
- current run rate
- required run rate

Powered by an XGBoost model trained on historical IPL ball-by-ball data.

---

## 🎥 Ball-by-Ball Match Replay
Replay historical IPL matches with:
- changing win probability
- momentum shifts
- turning point detection
- event-based commentary
- animated progression

---

## 📊 Momentum Analytics
Analyze:
- biggest momentum swings
- pressure overs
- batting dominance
- bowling control
- match turning points

---

## 🧠 Explainable AI with SHAP
Visualize why the model predicts a team will win or lose using SHAP feature explanations.

The app highlights:
- feature importance
- pressure indicators
- impact of wickets
- required run rate pressure

---

## 🎨 IPL Broadcast-Style UI
Features:
- IPL team logos
- team-based color themes
- dark analytics dashboard
- interactive charts
- animated replay visuals

---

## ⚡ Animated Match Center
Interactive replay engine showing:
- live-style score progression
- win probability changes
- wickets, fours, and sixes
- momentum transitions

---

# 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core Development |
| Streamlit | Web Application |
| XGBoost | Machine Learning |
| SHAP | Explainable AI |
| Plotly | Interactive Visualizations |
| Pandas | Data Processing |
| Scikit-learn | ML Pipeline |

---

# 📊 Model Evaluation

The XGBoost model was trained on historical IPL second-innings chase data using engineered match-state features.

| Metric | Score |
|---|---|
| Accuracy | 90.3% |
| Precision | 90% |
| Recall | 91% |
| F1-Score | 90% |

The model performs well in capturing:
- match pressure situations
- wicket impact
- required run rate dynamics
- momentum shifts during run chases

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
│
├── images/
│   └── logos/
│
├── models/
│   ├── xgboost_win_predictor.pkl
│   ├── random_forest_win_predictor.pkl
│
├── utils/
│   ├── charts.py
│   ├── commentary.py
│   ├── prediction.py
│   ├── replay.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore

