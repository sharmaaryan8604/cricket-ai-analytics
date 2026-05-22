import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

import pickle

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/ipl_win_prediction_dataset.csv"
)

print(df.head())

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
    'city'
]

numerical_cols = [
    'runs_remaining',
    'balls_remaining',
    'wickets_remaining',
    'current_run_rate',
    'required_run_rate'
]

# =====================================================
# PREPROCESSING
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
# MODEL PIPELINE
# =====================================================

model = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', LogisticRegression(
        max_iter=1000
    ))
])

# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# ACCURACY
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n")
print("=" * 50)
print("MODEL ACCURACY")
print("=" * 50)

print(f"Accuracy: {accuracy:.4f}")

# =====================================================
# SAVE MODEL
# =====================================================

with open(
    "models/win_prediction_model.pkl",
    "wb"
) as f:

    pickle.dump(model, f)

print("\nModel saved successfully!")