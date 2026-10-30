import datetime
import json
import os
import re

import pandas as pd
from nsetools import Nse

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def visible_len(s):
    """
    Input Structure:
    s (str): Input string containing ANSI codes.

    Who calls this function:
    Internal helper for print_table.

    Task:
    Calculate the visible length of a string by stripping ANSI codes.

    Code Logic:
    1. Define regex for ANSI codes.
    2. Substitute ANSI codes with empty string.
    3. Return length of result.

    Output:
    int: Visible length of the string.
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', str(s)))


def print_table(data, headers):
    """
    Input Structure:
    data (list of dict): Row data.
    headers (list of str): Column headers.

    Who calls this function:
    test_prediction_accuracy.

    Task:
    Print a properly aligned table capable of handling ANSI color codes.

    Code Logic:
    1. Calculate maximum width for each column based on headers and data (using visible_len).
    2. Add a buffer to widths.
    3. Print header row using calculated widths.
    4. Print separator row.
    5. Iterate through data and print each row, padding values based on visible_len.

    Output:
    None.
    """
    if not data:
        return

    # Calculate column widths
    col_widths = {h: len(h) for h in headers}
    for row in data:
        for h in headers:
            val = str(row.get(h, ""))
            v_len = visible_len(val)
            if v_len > col_widths[h]:
                col_widths[h] = v_len

    # Add padding buffer
    for h in col_widths:
        col_widths[h] += 2

    # Print Header
    header_str = "".join(h.ljust(col_widths[h]) for h in headers)
    print(header_str)
    print("-" * len(header_str))

    # Print Rows
    for row in data:
        row_str = ""
        for h in headers:
            val = str(row.get(h, ""))
            v_len = visible_len(val)
            padding = col_widths[h] - v_len
            row_str += val + " " * padding
        print(row_str)


def test_prediction_accuracy(df, filename="predictionData.json"):
    """
    Input Structure:
    df (pandas.DataFrame): Today's market data with 'symbol' and 'ltp' columns.
    filename (str): JSON file where past predictions are saved.

    Who calls this function:
    Main script when the -t or --test flag is provided.

    Task:
    Compare the most recent saved prediction (excluding today) against today's actual prices, print a color-coded performance table with hit rate and investment summary, and save each actual return back so the predictor calibrates itself.

    Code Logic:
    1. Load prediction history and pick the most recent date other than today.
    2. Map today's symbols to their ltp, falling back to a live quote for stocks that left the gainers list.
    3. Compute each pick's actual return from its saved entry price.
    4. Print the recommendation and performance tables using print_table.
    5. Print the hit rate and the 100,000 inr equal-weight investment summary.
    6. Save actual_return back into the prediction file.

    Output:
    pandas.DataFrame: The dataframe (unmodified).
    """
    if df.empty or not os.path.exists(filename):
        print("Nothing to test. Save predictions with -p first.")
        return df

    with open(filename, "r") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: could not parse {filename}.")
            return df

    today = datetime.datetime.now().strftime("%d-%m-%Y")
    past = [d for d in history if d != today]
    if not past:
        print("No past predictions to test. Run -p today and -t tomorrow.")
        return df

    date = max(past, key=lambda d: datetime.datetime.strptime(d, "%d-%m-%Y"))
    predictions = history[date]

    df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")
    prices = dict(zip(df["symbol"], df["ltp"]))
    nse = Nse()

    print(f"\n----- Predictions from {date} vs Today's Actual -----")

    rec_data = []
    for rank, (symbol, pred) in enumerate(predictions.items(), 1):
        rec_data.append({
            "#": str(rank),
            "Stock": symbol,
            "Predicted Change": f"+{pred['Predicted_Return']:.2f}%",
            "Entry Price": str(pred["ltp"]),
        })

    print_table(rec_data, ["#", "Stock", "Predicted Change", "Entry Price"])
    print("\n----- How they performed today -----")

    perf_data = []
    for rank, (symbol, pred) in enumerate(predictions.items(), 1):
        actual = prices.get(symbol)
        if pd.isna(actual):
            try:
                actual = nse.get_quote(symbol).get("lastPrice")
            except Exception:
                actual = None

        if actual is None:
            perf_data.append({"#": str(rank), "Stock": symbol, "Actual Change": "N/A", "Closing": "N/A"})
            continue

        change = (actual - pred["ltp"]) / pred["ltp"] * 100
        pred["actual_return"] = round(change, 2)

        color = GREEN if change >= 2 else YELLOW if change >= 0 else RED
        sign = "+" if change >= 0 else ""
        perf_data.append({
            "#": str(rank),
            "Stock": symbol,
            "Actual Change": f"{color}{sign}{change:.2f}%{RESET}",
            "Closing": f"{color}{actual}{RESET}",
        })

    print_table(perf_data, ["#", "Stock", "Actual Change", "Closing"])

    returns = [v["actual_return"] for v in predictions.values() if "actual_return" in v]
    if returns:
        mean = sum(returns) / len(returns)
        hit_rate = 100 * sum(r > 0 for r in returns) / len(returns)
        total = 100000 * (1 + mean / 100)
        profit = total - 100000
        pl_color = GREEN if profit >= 0 else RED
        sign = "+" if profit >= 0 else "-"

        print(f"\n{pl_color}If you invested 100,000 inr equally in these {len(returns)} stocks, you would now have {int(total):,} inr. ({sign} {abs(int(profit)):,}){RESET}")
        print(f"Hit rate: {hit_rate:.0f}% positive. Average return: {mean:+.2f}%")

        with open(filename, "w") as f:
            json.dump(history, f, indent=4)

        print(f"\nActual returns saved to {filename} to calibrate future predictions.")

    return df
