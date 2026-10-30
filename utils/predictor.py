import json
import os

import pandas as pd


def expected_return(filename="predictionData.json"):
    """
    Input Structure:
    filename (str): JSON file storing past predictions and their actual returns.

    Who calls this function:
    predict_future_values, to calibrate Predicted_Return with real history.

    Task:
    Compute the strategy's average actual next-day return from past predictions, falling back to 2.0 when no history exists yet.

    Code Logic:
    1. Return 2.0 if the file does not exist.
    2. Load the history and collect every recorded actual_return.
    3. Return their mean, or 2.0 if none were recorded yet.

    Output:
    float: The strategy's historical average next-day return in percent.
    """
    if not os.path.exists(filename):
        return 2.0

    with open(filename, "r") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            return 2.0

    actuals = [row["actual_return"] for day in history.values() for row in day.values() if "actual_return" in row]
    return sum(actuals) / len(actuals) if actuals else 2.0


def predict_future_values(df, filename="predictionData.json"):
    """
    Input Structure:
    df (pandas.DataFrame): A DataFrame containing stock market data with columns like 'ltp', 'open_price', 'high_price', 'low_price', 'prev_price', 'net_price', and 'turnover'.

    Who calls this function:
    Main script when the -p or --predict flag is used.

    Task:
    Predict tomorrow's percentage change using a momentum strategy: score stocks on intraday momentum, price position and volume, filter out weak-volume movers, and calibrate the score against the strategy's own recorded performance.

    Code Logic:
    1. Check if the DataFrame is empty.
    2. Convert required columns to numeric format.
    3. Drop stocks below the median turnover so only strong-volume movers are recommended.
    4. Compute intraday return using ltp and open_price.
    5. Compute price position within daily range.
    6. Compute previous return using prev_price.
    7. Normalize turnover to represent volume strength.
    8. Combine all features into a weighted momentum score and z-score it.
    9. Scale the z-score by expected_return so 1 std above average predicts the strategy's historical average gain.
    10. Return the updated DataFrame with 'Predicted_Return' and 'Predicted_Value'.

    Output:
    pandas.DataFrame: DataFrame enriched with 'Predicted_Return' and 'Predicted_Value'.
    """
    if df.empty:
        return df

    df = df.copy()

    numeric_cols = [
        "ltp",
        "open_price",
        "high_price",
        "low_price",
        "prev_price",
        "net_price",
        "turnover",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["turnover"] >= df["turnover"].median()]

    df["intraday_return"] = (df["ltp"] - df["open_price"]) / df["open_price"]
    df["range"] = df["high_price"] - df["low_price"]
    df["price_position"] = (df["ltp"] - df["low_price"]) / df["range"].replace(0, 1)
    df["prev_return"] = (df["ltp"] - df["prev_price"]) / df["prev_price"]
    df["turnover_norm"] = df["turnover"] / df["turnover"].max()

    df["Momentum_Score"] = (
        0.4 * df["intraday_return"]
        + 0.3 * df["price_position"]
        + 0.2 * df["prev_return"]
        + 0.1 * df["turnover_norm"]
    )

    df["Momentum_Score"] = (df["Momentum_Score"] - df["Momentum_Score"].mean()) / df["Momentum_Score"].std()

    df["Predicted_Return"] = df["Momentum_Score"] * expected_return(filename)

    df["Predicted_Value"] = df["ltp"] * (1 + df["Predicted_Return"] / 100)

    return df
