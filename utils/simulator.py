import json
import os
import tempfile

import pandas as pd
import yfinance as yf

from utils import predictor

UNIVERSE = (
    "ADANIENT", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL",
    "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "ULTRACEMCO",
    "WIPRO", "ADANIPORTS", "LTIM",
)


def simulate(days):
    """
    Input Structure:
    days (int): Number of trading days to walk the strategy forward over.

    Who calls this function:
    Main script when the -s or --simulate flag is used.

    Task:
    Replay the daily loop (predict top 5, grade against the next day, recalibrate) over historical NIFTY 50 data and report how the strategy would have performed over that span.

    Code Logic:
    1. Download daily candles for the NIFTY 50 universe and the NIFTY index.
    2. Walk forward one trading day at a time, skipping the first day (no previous close) and the last day (no next day to grade against).
    3. Rebuild each day's market snapshot in the live schema and run predict_future_values with a temporary history file.
    4. Record the top 5 picks, grade them against the next day's close, and save actual returns so later days calibrate exactly like the real loop.
    5. Print hit rate, average pick return, the compounded 100,000 inr portfolio value and the NIFTY buy-and-hold comparison.

    Output:
    None (prints the simulation summary).
    """
    print(f"Downloading {days} trading days of NIFTY 50 data...")
    data = yf.download(" ".join(s + ".NS" for s in UNIVERSE) + " ^NSEI",
                       period=f"{days * 2 + 10}d", group_by="ticker", auto_adjust=True, progress=False)
    if data.empty:
        print("No data received. Exiting.")
        return

    close = pd.DataFrame({s: data[s + ".NS"]["Close"] for s in UNIVERSE})
    prev_close = close.shift(1)
    candles = {c: pd.DataFrame({s: data[s + ".NS"][c] for s in UNIVERSE}) for c in ("Open", "High", "Low", "Volume")}

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    history = {}
    day_results = []
    dates = data.index[-(days + 2):]

    for t in range(1, len(dates) - 1):
        rows = []
        for s in UNIVERSE:
            ltp, prev = close.at[dates[t], s], prev_close.at[dates[t], s]
            if pd.isna(ltp) or pd.isna(prev) or prev == 0:
                continue
            rows.append({
                "symbol": s,
                "ltp": ltp,
                "open_price": candles["Open"].at[dates[t], s],
                "high_price": candles["High"].at[dates[t], s],
                "low_price": candles["Low"].at[dates[t], s],
                "prev_price": prev,
                "net_price": round((ltp - prev) / prev * 100, 2),
                "turnover": candles["Volume"].at[dates[t], s] * ltp,
            })

        df = predictor.predict_future_values(pd.DataFrame(rows), filename=tmp)
        picks = {}
        for _, row in df.sort_values("Predicted_Return", ascending=False).head(5).iterrows():
            nxt = close.at[dates[t + 1], row["symbol"]]
            if pd.isna(nxt):
                continue
            picks[row["symbol"]] = {
                "Predicted_Return": row["Predicted_Return"],
                "ltp": row["ltp"],
                "actual_return": round((nxt - row["ltp"]) / row["ltp"] * 100, 2),
            }

        history[dates[t].strftime("%d-%m-%Y")] = picks
        with open(tmp, "w") as f:
            json.dump(history, f)
        if picks:
            day_results.append([p["actual_return"] for p in picks.values()])

    os.remove(tmp)

    flat = [r for day in day_results for r in day]
    if not flat:
        print("No days could be simulated.")
        return

    value = 100000.0
    for day in day_results:
        value *= sum(1 + r / 100 for r in day) / len(day)

    nifty = data["^NSEI"]["Close"].reindex(dates).dropna()
    benchmark = (nifty.iloc[-1] / nifty.iloc[0] - 1) * 100 if len(nifty) > 1 else 0.0

    print("\n----- Simulation Summary -----")
    print(f"Days simulated: {len(day_results)}, picks graded: {len(flat)}")
    print(f"Hit rate: {100 * sum(r > 0 for r in flat) / len(flat):.0f}% positive picks")
    print(f"Average pick return: {sum(flat) / len(flat):+.2f}%/day (best {max(flat):+.2f}%, worst {min(flat):+.2f}%)")
    pl = "+" if value >= 100000 else "-"
    print(f"100,000 inr compounded to {int(value):,} inr ({pl} {abs(value - 100000):,.0f} inr, {(value / 100000 - 1) * 100:+.1f}%)")
    print(f"NIFTY 50 buy & hold over the same span: {benchmark:+.2f}%")
