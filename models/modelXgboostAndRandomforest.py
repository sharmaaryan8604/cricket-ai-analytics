import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/ipl_win_prediction_dataset.csv"
)

print("=" * 50)
print("DATASET LOADED")
print("=" * 50)

print(df.head())

# =====================================================
# ADD ADVANCED FEATURES
# =====================================================

# -----------------------------------------------------
# LAST 30 BALL RUNS
# -----------------------------------------------------

df['last_30_runs'] = df.groupby(
    ['batting_team']
)['runs_remaining'].rolling(
    30,
    min_periods=1
).mean().reset_index(
    level=0,
    drop=True
)

# -----------------------------------------------------
# LAST 30 BALL WICKETS
# -----------------------------------------------------

df['last_30_wickets'] = df.groupby(
    ['batting_team']
)['wickets_remaining'].rolling(
    30,
    min_periods=1
).mean().reset_index(
    level=0,
    drop=True
)

# -----------------------------------------------------
# DOT BALL PRESSURE
# -----------------------------------------------------

df['last_30_dot_balls'] = (
    df['required_run_rate'] >
    df['current_run_rate']
).astype(int)

# -----------------------------------------------------
# BOUNDARY MOMENTUM
# -----------------------------------------------------

df['last_30_boundaries'] = (
    df['current_run_rate'] * 2
).round()

# -----------------------------------------------------
# PHASE FEATURE
# -----------------------------------------------------

def get_phase(balls_remaining):

    if balls_remaining > 84:
        return 'Powerplay'

    elif balls_remaining > 30:
        return 'Middle'

    return 'Death'

df['phase'] = df[
    'balls_remaining'
].apply(get_phase)

# =====================================================
# CLEAN DATA
# =====================================================

df = df.replace(
    [float('inf'), -float('inf')],
    0
)

df = df.dropna()

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop(
    'result',
    axis=1
)

y = df['result']

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# CATEGORICAL & NUMERICAL COLUMNS
# =====================================================

categorical_cols = [
    'batting_team',
    'bowling_team',
    'city',
    'phase'
]

numerical_cols = [
    'runs_remaining',
    'balls_remaining',
    'wickets_remaining',

    'current_run_rate',
    'required_run_rate',

    'last_30_runs',
    'last_30_wickets',
    'last_30_dot_balls',
    'last_30_boundaries'
]

# =====================================================
# PREPROCESSOR
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_cols
        )
    ],
    remainder='passthrough'
)

# =====================================================
# RANDOM FOREST MODEL
# =====================================================

rf_model = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# =====================================================
# TRAIN RANDOM FOREST
# =====================================================

print("\n")
print("=" * 50)
print("TRAINING RANDOM FOREST")
print("=" * 50)

rf_model.fit(
    X_train,
    y_train
)

# =====================================================
# RANDOM FOREST PREDICTIONS
# =====================================================

rf_pred = rf_model.predict(X_test)

# =====================================================
# RANDOM FOREST ACCURACY
# =====================================================

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("\n")
print("=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)

print(f"Accuracy: {rf_accuracy:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    rf_pred
))

print("\nConfusion Matrix:\n")

print(confusion_matrix(
    y_test,
    rf_pred
))

# =====================================================
# SAVE RANDOM FOREST MODEL
# =====================================================

with open(
    "models/random_forest_win_predictor.pkl",
    "wb"
) as f:

    pickle.dump(rf_model, f)

print("\nRandom Forest model saved!")

# =====================================================
# XGBOOST MODEL
# =====================================================

xgb_model = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=1,
        min_child_weight=3,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    ))
])

# =====================================================
# TRAIN XGBOOST
# =====================================================

print("\n")
print("=" * 50)
print("TRAINING XGBOOST")
print("=" * 50)

xgb_model.fit(
    X_train,
    y_train
)

# =====================================================
# XGBOOST PREDICTIONS
# =====================================================

xgb_pred = xgb_model.predict(X_test)

# =====================================================
# XGBOOST ACCURACY
# =====================================================

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)

print("\n")
print("=" * 50)
print("XGBOOST RESULTS")
print("=" * 50)

print(f"Accuracy: {xgb_accuracy:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    xgb_pred
))

print("\nConfusion Matrix:\n")

print(confusion_matrix(
    y_test,
    xgb_pred
))

# =====================================================
# SAVE XGBOOST MODEL
# =====================================================

with open(
    "models/xgboost_win_predictor.pkl",
    "wb"
) as f:

    pickle.dump(xgb_model, f)

print("\nXGBoost model saved!")

# =====================================================
# FINAL COMPARISON
# =====================================================

print("\n")
print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(f"Random Forest Accuracy : {rf_accuracy:.4f}")
print(f"XGBoost Accuracy       : {xgb_accuracy:.4f}")

# =====================================================
# BEST MODEL
# =====================================================

if xgb_accuracy > rf_accuracy:

    print("\nBest Model: XGBoost")

else:

    print("\nBest Model: Random Forest")