import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/ipl_win_prediction_dataset.csv"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = pickle.load(
    open(
        "models/xgboost_win_predictor.pkl",
        "rb"
    )
)

# =====================================================
# FEATURES
# =====================================================

X = df.drop(
    'result',
    axis=1
)

sample = X.sample(
    1000,
    random_state=42
)

# =====================================================
# GET PIPELINE PARTS
# =====================================================

preprocessor = model.named_steps[
    'preprocessor'
]

xgb_model = model.named_steps[
    'classifier'
]

# =====================================================
# TRANSFORM DATA
# =====================================================

X_processed = preprocessor.transform(
    sample
)

# =====================================================
# GET FEATURE NAMES
# =====================================================

feature_names = preprocessor.get_feature_names_out()

# =====================================================
# SHAP EXPLAINER
# =====================================================

explainer = shap.TreeExplainer(
    xgb_model
)

shap_values = explainer.shap_values(
    X_processed
)

# =====================================================
# SUMMARY PLOT
# =====================================================

shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=feature_names,
    show=False
)

plt.title(
    "SHAP Feature Importance - IPL Win Predictor"
)

plt.tight_layout()

plt.show()