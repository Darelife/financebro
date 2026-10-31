import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from utils import features, predictor

UNIVERSE = (
    "ADANIENT", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL",
    "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "ULTRACEMCO",
    "WIPRO", "ADANIPORTS", "LTIM",
)

BENCHMARKS = {"^NSEI": "NIFTY 50", "^NSMIDCP": "NIFTY Next 50"}


def _fetch():
    data = yf.download(" ".join(s + ".NS" for s in UNIVERSE) + " " + " ".join(BENCHMARKS),
                       period="3y", group_by="ticker", auto_adjust=True, progress=False)
    if data.empty:
        return None
    close = pd.DataFrame({s: data[s + ".NS"]["Close"] for s in UNIVERSE})
    volume = pd.DataFrame({s: data[s + ".NS"]["Volume"] for s in UNIVERSE})
    bench = {b: data[b]["Close"] for b in BENCHMARKS}
    return close, volume, bench


def simulate(days):
    """
    Input Structure:
    days (int): Number of trading days to walk the strategy forward over.

    Who calls this function:
    Main script when the -s or --simulate flag is used.

    Task:
    Walk every strategy sleeve forward day by day over historical NIFTY 50 data, grade each sleeve's own top 5 picks against the next day, and report per-sleeve results plus NIFTY benchmarks in a table and a cumulative-return chart.

    Code Logic:
    1. Download 3 years of daily candles for the universe and the two benchmark indices.
    2. Build the feature panel once.
    3. Walk forward one trading day at a time over the requested span, running the predictor on each day.
    4. For each sleeve (and the ensemble), take its own top 5 picks, grade them against the next day's close, and compound the sleeve's daily mean return into a cumulative curve.
    5. Compound each benchmark index alongside.
    6. Print a per-sleeve summary table and save the cumulative comparison chart to strategy_comparison.png.

    Output:
    None (prints the summary and saves strategy_comparison.png).
    """
    print("Downloading 3 years of NIFTY 50 data...")
    fetched = _fetch()
    if fetched is None:
        print("No data received. Exiting.")
        return
    close, volume, bench = fetched
    dates = close.index
    start = max(len(dates) - days - 1, 60)

    bench_cum = {b: (bench[b].pct_change(fill_method=None).fillna(0) + 1).cumprod().reindex(dates).ffill()
                 for b in BENCHMARKS}
    bench_cum = {b: c / c.iloc[start] * 100 for b, c in bench_cum.items()}

    panel = features.build_features(close, volume)

    curves = {name: [100.0] for name in predictor.WEIGHTS}
    curves["ensemble"] = [100.0]
    bench_curves = {b: [100.0] for b in BENCHMARKS}
    returns = {name: [] for name in curves}
    sim_dates = []

    for t in range(start, len(dates) - 1):
        df = predictor.predict_future_values(panel, t)
        if df.empty:
            continue

        for name in curves:
            picks = df.nlargest(5, "Predicted_Return" if name == "ensemble" else name)
            day = [(close.at[dates[t + 1], s] - ltp) / ltp * 100
                   for s, ltp in zip(picks["symbol"], picks["ltp"])
                   if not pd.isna(close.at[dates[t + 1], s])]
            if not day:
                continue
            returns[name].extend(day)
            curves[name].append(curves[name][-1] * (1 + sum(day) / len(day) / 100))

        for b in BENCHMARKS:
            bench_curves[b].append(bench_cum[b].at[dates[t + 1]])

        sim_dates.append(dates[t + 1])

    if not sim_dates:
        print("No days could be simulated.")
        return

    print(f"\n----- Simulation Summary ({len(sim_dates)} days) -----")
    print(f"{'Strategy':<12} {'Hit rate':>9} {'Avg/day':>9} {'Total':>9}")
    for name, curve in curves.items():
        rs = returns[name]
        hit = 100 * sum(r > 0 for r in rs) / len(rs) if rs else 0
        avg = sum(rs) / len(rs) if rs else 0
        total = curve[-1] - 100
        print(f"{name:<12} {hit:>8.0f}% {avg:>+8.2f}% {total:>+8.1f}%")

    for b in BENCHMARKS:
        print(f"{BENCHMARKS[b]:<12} {'':>9} {'':>9} {bench_curves[b][-1] - 100:>+8.1f}%")

    plt.figure(figsize=(12, 6))
    for name, curve in curves.items():
        plt.plot(sim_dates[:len(curve) - 1], curve[1:], label=name)
    for b in BENCHMARKS:
        plt.plot(sim_dates[:len(bench_curves[b]) - 1], bench_curves[b][1:], "--", label=BENCHMARKS[b])
    plt.title("Strategy comparison (top-5 daily picks, NIFTY 50 universe)")
    plt.ylabel("Growth of 100 inr (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("strategy_comparison.png", dpi=120, bbox_inches="tight")
    print("\nChart saved to strategy_comparison.png")
