import numpy as np
import pandas as pd

FEATURES = ["ret1", "ret5", "ret20", "vol20", "dist_ma20", "rsi14", "vol_ratio"]


def build_features(close, volume):
    """
    Input Structure:
    close (pandas.DataFrame): Daily closing prices, dates as rows and symbols as columns.
    volume (pandas.DataFrame): Daily volumes matching the close DataFrame.

    Who calls this function:
    simulator.simulate and the -p path in the main script.

    Task:
    Build the feature panel every strategy scores on: per-date, per-symbol momentum, volatility and volume features.

    Code Logic:
    1. Compute daily returns, Wilder-smoothed RSI gains and losses.
    2. Derive 1/5/20-day returns, 20-day return volatility, distance from the 20-day mean and volume ratio.
    3. Return a dict of DataFrames (each dates x symbols) keyed by feature name, plus the close itself.

    Output:
    dict: Feature DataFrames keyed by name, with "close" included.
    """
    ret = close.pct_change(fill_method=None)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()

    return {
        "close": close,
        "ret1": ret,
        "ret5": close.pct_change(5, fill_method=None),
        "ret20": close.pct_change(20, fill_method=None),
        "vol20": ret.rolling(20).std(),
        "dist_ma20": close / close.rolling(20).mean() - 1,
        "rsi14": 100 - 100 / (1 + gain / loss.replace(0, np.nan)),
        "vol_ratio": volume / volume.rolling(20).mean(),
    }
