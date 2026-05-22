def build_momentum_drivers(
    batting_team,
    bowling_team,
    pressure_index,
    current_run_rate,
    required_run_rate,
    last_30_boundaries,
    last_30_dot_balls,
    wickets_remaining,
):

    momentum_drivers = []

    if current_run_rate >= required_run_rate:

        momentum_drivers.append(
            f"{batting_team} is ahead of the asking rate by {abs(pressure_index):.2f} runs per over."
        )

    else:

        momentum_drivers.append(
            f"{bowling_team} is squeezing the chase with a run-rate gap of {abs(pressure_index):.2f}."
        )

    if last_30_boundaries >= 4:

        momentum_drivers.append(
            "Recent boundary frequency suggests attacking intent from the batting side."
        )

    if last_30_dot_balls >= 10:

        momentum_drivers.append(
            "Dot-ball pressure is building and slowing the chase."
        )

    if wickets_remaining <= 3:

        momentum_drivers.append(
            "Limited wickets in hand make the next over especially high leverage."
        )

    return momentum_drivers


def build_ai_insights(
    current_run_rate,
    required_run_rate,
    wickets_remaining,
    balls_remaining,
    last_30_dot_balls,
    last_30_boundaries,
):

    insights = []

    if required_run_rate > current_run_rate:

        insights.append(
            "Required run rate is higher than current scoring rate."
        )

    if wickets_remaining <= 3:

        insights.append(
            "Very few wickets remaining increases pressure."
        )

    if balls_remaining < 24:

        insights.append(
            "Death overs increase volatility."
        )

    if current_run_rate >= 10:

        insights.append(
            "Batting side is scoring aggressively."
        )

    if last_30_dot_balls >= 12:

        insights.append(
            "Too many dot balls are slowing momentum."
        )

    if last_30_boundaries >= 6:

        insights.append(
            "Frequent boundaries are boosting momentum."
        )

    return insights


def build_live_commentary(match_row, current_ball):

    runs = match_row["batter_runs"]

    if match_row["wicket"] == 1:

        return (
            f"{current_ball} WICKET! "
            f"{match_row['player_out']} is out!"
        )

    if runs == 6:

        return (
            f"{current_ball} SIX! "
            f"Huge hit by {match_row['batter']}!"
        )

    if runs == 4:

        return (
            f"{current_ball} FOUR! "
            f"Beautiful boundary by {match_row['batter']}!"
        )

    if runs == 0:

        return f"{current_ball} Dot ball."

    return f"{current_ball} {runs} run(s) taken."
