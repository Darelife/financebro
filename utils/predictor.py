import pandas as pd

from utils import strategies

WEIGHTS = {"momentum": 0.3, "reversion": 0.2, "probability": 0.15, "ml": 0.35}


def predict_future_values(panel, t):
    """
    Input Structure:
    panel (dict): Feature DataFrames from features.build_features, each dates x symbols.
    t (int): Row index (position) of the prediction date within the panel.

    Who calls this function:
    simulator.simulate and the -p path in the main script.

    Task:
    Combine the four strategy sleeves into one ensemble ranking and predict tomorrow's percentage return for each stock.

    Code Logic:
    1. Compute every sleeve's score for date t and z-score each cross-sectionally.
    2. Combine them with fixed weights into an ensemble score.
    3. Scale the ensemble by the universe's mean absolute daily return over the trailing 20 days.
    4. Drop stocks in the bottom volatility tercile and stocks with missing scores.
    5. Return one row per remaining symbol with ltp, the predicted return and each sleeve's z-score.

    Output:
    pandas.DataFrame: Columns symbol, ltp, Predicted_Return, momentum, reversion, probability, ml.
    """
    ret1 = panel["ret1"]
    scores = {
        "momentum": strategies.momentum(panel, t),
        "reversion": strategies.reversion(panel, t),
        "probability": strategies.probability(panel, t, ret1),
        "ml": strategies.ml(panel, t, ret1),
    }

    zs = {name: strategies.zcross(s) for name, s in scores.items()}
    ensemble = sum(WEIGHTS[name] * zs[name] for name in WEIGHTS)

    scale = ret1.iloc[max(0, t - 20):t].abs().mean().mean() * 100
    vol20 = panel["vol20"].iloc[t]
    keep = (vol20 > vol20.quantile(1 / 3)) & ensemble.notna()

    df = pd.DataFrame({
        "symbol": ensemble.index,
        "ltp": panel["close"].iloc[t].values,
        "Predicted_Return": (ensemble * scale).values,
    })
    for name in WEIGHTS:
        df[name] = zs[name].values

    return df[keep.values].sort_values("Predicted_Return", ascending=False)
