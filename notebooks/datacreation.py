import json
import pandas as pd
import os

# =====================================================
# IPL JSON DATASET EXTRACTION + FEATURE ENGINEERING
# =====================================================

# Path containing all IPL json files
data_path = "data/ipl_json"

# Get all files
files = os.listdir(data_path)

# Store all deliveries
all_rows = []

# =====================================================
# LOOP THROUGH FILES
# =====================================================

for file in files:

    # Skip non-json files
    if not file.endswith(".json"):
        continue

    file_path = os.path.join(data_path, file)

    # Load JSON
    with open(file_path, "r") as f:
        data = json.load(f)

    # =====================================================
    # MATCH INFO
    # =====================================================

    match_id = file.replace(".json", "")

    info = data.get("info", {})

    city = info.get("city", "Unknown")

    venue = info.get("venue", "Unknown")

    dates = info.get("dates", ["Unknown"])

    match_date = dates[0]

    teams = info.get("teams", [])

    # Match winner
    winner = info.get("outcome", {}).get(
        "winner",
        "No Result"
    )

    # =====================================================
    # PROCESS INNINGS
    # =====================================================

    for innings_number, inning in enumerate(
        data["innings"],
        start=1
    ):

        batting_team = inning.get(
            "team",
            "Unknown"
        )

        # Find bowling team
        bowling_team = [
            team for team in teams
            if team != batting_team
        ]

        bowling_team = (
            bowling_team[0]
            if bowling_team else "Unknown"
        )

        # =====================================================
        # PROCESS OVERS
        # =====================================================

        for over in inning["overs"]:

            over_number = over["over"]

            # =====================================================
            # PROCESS DELIVERIES
            # =====================================================

            for ball_number, delivery in enumerate(
                over["deliveries"],
                start=1
            ):

                batter = delivery.get(
                    "batter",
                    None
                )

                bowler = delivery.get(
                    "bowler",
                    None
                )

                non_striker = delivery.get(
                    "non_striker",
                    None
                )

                # =====================================================
                # RUNS
                # =====================================================

                runs_data = delivery.get(
                    "runs",
                    {}
                )

                batter_runs = runs_data.get(
                    "batter",
                    0
                )

                extra_runs = runs_data.get(
                    "extras",
                    0
                )

                total_runs = runs_data.get(
                    "total",
                    0
                )

                # =====================================================
                # EXTRAS
                # =====================================================

                extras_data = delivery.get(
                    "extras",
                    {}
                )

                wides = extras_data.get(
                    "wides",
                    0
                )

                noballs = extras_data.get(
                    "noballs",
                    0
                )

                byes = extras_data.get(
                    "byes",
                    0
                )

                legbyes = extras_data.get(
                    "legbyes",
                    0
                )

                # =====================================================
                # WICKETS
                # =====================================================

                wicket = 0

                player_out = None

                if "wickets" in delivery:

                    wicket = 1

                    player_out = delivery[
                        "wickets"
                    ][0].get(
                        "player_out",
                        None
                    )

                # =====================================================
                # CREATE ROW
                # =====================================================

                row = {

                    # Match Info
                    "match_id": match_id,
                    "city": city,
                    "venue": venue,
                    "date": match_date,
                    "winner": winner,

                    # Innings
                    "innings": innings_number,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,

                    # Ball Info
                    "over": over_number,
                    "ball": ball_number,

                    # Players
                    "batter": batter,
                    "bowler": bowler,
                    "non_striker": non_striker,

                    # Runs
                    "batter_runs": batter_runs,
                    "extra_runs": extra_runs,
                    "total_runs": total_runs,

                    # Extras
                    "wides": wides,
                    "noballs": noballs,
                    "byes": byes,
                    "legbyes": legbyes,

                    # Wickets
                    "wicket": wicket,
                    "player_out": player_out
                }

                all_rows.append(row)

# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(all_rows)

print("Dataset Shape:", df.shape)

print(df.head())

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# -----------------------------------------------------
# LEGAL BALL
# -----------------------------------------------------

df["legal_ball"] = (
    df["wides"] == 0
).astype(int)

# -----------------------------------------------------
# TEAM SCORE
# -----------------------------------------------------

df["team_score"] = df.groupby(
    ["match_id", "innings"]
)["total_runs"].cumsum()

# -----------------------------------------------------
# WICKETS FALLEN
# -----------------------------------------------------

df["wickets_fallen"] = df.groupby(
    ["match_id", "innings"]
)["wicket"].cumsum()

# -----------------------------------------------------
# BATTER SCORE
# -----------------------------------------------------

df["batter_score"] = df.groupby(
    ["match_id", "innings", "batter"]
)["batter_runs"].cumsum()

# -----------------------------------------------------
# BALLS FACED
# -----------------------------------------------------

df["balls_faced"] = df.groupby(
    ["match_id", "innings", "batter"]
)["legal_ball"].cumsum()

# -----------------------------------------------------
# STRIKE RATE
# -----------------------------------------------------

df["strike_rate"] = (
    df["batter_score"] * 100 /
    df["balls_faced"].replace(0, 1)
)

# -----------------------------------------------------
# BOWLER RUNS CONCEDED
# -----------------------------------------------------

df["bowler_runs_conceded"] = df.groupby(
    ["match_id", "innings", "bowler"]
)["total_runs"].cumsum()

# -----------------------------------------------------
# BOWLER WICKETS
# -----------------------------------------------------

df["bowler_wickets"] = df.groupby(
    ["match_id", "innings", "bowler"]
)["wicket"].cumsum()

# -----------------------------------------------------
# BALLS BOWLED
# -----------------------------------------------------

df["balls_bowled"] = df.groupby(
    ["match_id", "innings", "bowler"]
)["legal_ball"].cumsum()

# -----------------------------------------------------
# ECONOMY
# -----------------------------------------------------

df["economy"] = (
    df["bowler_runs_conceded"] * 6 /
    df["balls_bowled"].replace(0, 1)
)

# -----------------------------------------------------
# BALLS ELAPSED
# -----------------------------------------------------

df["balls_elapsed"] = df.groupby(
    ["match_id", "innings"]
)["legal_ball"].cumsum()

# -----------------------------------------------------
# CURRENT RUN RATE
# -----------------------------------------------------

df["current_run_rate"] = (
    df["team_score"] * 6 /
    df["balls_elapsed"].replace(0, 1)
)

# =====================================================
# MOMENTUM FEATURES
# =====================================================

# Last 30 ball runs

df['last_30_runs'] = df.groupby(
    ['match_id', 'innings']
)['total_runs'].rolling(
    30,
    min_periods=1
).sum().reset_index(
    level=[0,1],
    drop=True
)

# Last 30 ball wickets

df['last_30_wickets'] = df.groupby(
    ['match_id', 'innings']
)['wicket'].rolling(
    30,
    min_periods=1
).sum().reset_index(
    level=[0,1],
    drop=True
)

# Dot balls

df['dot_ball'] = (
    df['total_runs'] == 0
).astype(int)

df['last_30_dot_balls'] = df.groupby(
    ['match_id', 'innings']
)['dot_ball'].rolling(
    30,
    min_periods=1
).sum().reset_index(
    level=[0,1],
    drop=True
)

# Boundaries

df['boundary'] = (
    df['batter_runs'].isin([4, 6])
).astype(int)

df['last_30_boundaries'] = df.groupby(
    ['match_id', 'innings']
)['boundary'].rolling(
    30,
    min_periods=1
).sum().reset_index(
    level=[0,1],
    drop=True
)

# Match phase

def get_phase(over):

    if over <= 5:
        return 'Powerplay'

    elif over <= 14:
        return 'Middle'

    return 'Death'

df['phase'] = df['over'].apply(get_phase)

# =====================================================
# MATCH TARGET
# =====================================================

first_innings = df[
    df["innings"] == 1
]

target_scores = first_innings.groupby(
    "match_id"
)["total_runs"].sum()

df["target"] = df["match_id"].map(
    target_scores + 1
)

# =====================================================
# CHASE FEATURES
# =====================================================

df["runs_remaining"] = (
    df["target"] - df["team_score"]
)

df["balls_remaining"] = (
    120 - df["balls_elapsed"]
)

df["wickets_remaining"] = (
    10 - df["wickets_fallen"]
)

df["required_run_rate"] = (
    df["runs_remaining"] * 6 /
    df["balls_remaining"].replace(0, 1)
)

# =====================================================
# RESULT COLUMN
# =====================================================

df["result"] = (
    df["batting_team"] == df["winner"]
).astype(int)

# =====================================================
# CREATE ML DATASET
# =====================================================

model_df = df[
    df["innings"] == 2
][[
    "batting_team",
    "bowling_team",
    "city",
    "phase",

    "runs_remaining",
    "balls_remaining",
    "wickets_remaining",

    "current_run_rate",
    "required_run_rate",

    "last_30_runs",
    "last_30_wickets",
    "last_30_dot_balls",
    "last_30_boundaries",

    "result"
]]

# =====================================================
# CLEAN DATA
# =====================================================

model_df = model_df.replace(
    [float('inf'), -float('inf')],
    0
)

model_df = model_df.dropna()

# =====================================================
# SAVE DATASETS
# =====================================================

os.makedirs("data", exist_ok=True)

# Save full ball-by-ball dataset
df.to_csv(
    "data/ipl_ball_by_ball.csv",
    index=False
)

# Save ML dataset
model_df.to_csv(
    "data/ipl_win_prediction_dataset.csv",
    index=False
)

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n")
print("=" * 50)
print("FINAL DATASET CREATED")
print("=" * 50)

print("\nBall-by-ball dataset shape:")
print(df.shape)

print("\nML dataset shape:")
print(model_df.shape)

print("\nColumns:")
print(model_df.columns)

print("\nSample Data:")
print(model_df.head())