import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from utils.features import FEATURES

# Next-day moves beyond this are news, not the pattern we want to learn.
CLIP = 3.0

# Bucket edges for the probability sleeve's empirical lookup.
BINS = [-0.1, -0.05, 0, 0.05, 0.1]


def zcross(series):
    return (series - series.mean()) / series.std()


# Trend persistence: stocks whose recent returns beat the market's tend to keep
# outperforming for a few more days. 5-day moves get more weight than 20-day ones
# because short-term momentum decays fast.
def momentum(panel, t):
    return 0.6 * zcross(panel["ret5"].iloc[t]) + 0.4 * zcross(panel["ret20"].iloc[t])


# Mean reversion: stocks stretched far above their 20-day average tend to snap back,
# so we short the stretch. Only applied at RSI extremes (<35 oversold, >65 overbought)
# where reversion is most reliable; mid-range names get no score at all.
def reversion(panel, t):
    rsi = panel["rsi14"].iloc[t]
    return -zcross(panel["dist_ma20"].iloc[t]).where((rsi < 35) | (rsi > 65))


# Empirical base rates: look up how often this kind of stock went up the next day.
# Every past day is binned by its 5-day return and distance from its 20-day average,
# and each (ret5, dist) bucket records its historical share of next-day-up outcomes.
# Laplace smoothing (+1/+2) keeps thin buckets near 50/50 instead of trusting them.
# Score maps the probability to -1..+1 around the 50% coin flip.
def probability(panel, t, ret1, window=250):
    lo = max(0, t - window)
    if t - lo < 20:
        return pd.Series(np.nan, index=ret1.columns)

    b5 = np.digitize(panel["ret5"].iloc[lo:t].values, BINS)
    bd = np.digitize(panel["dist_ma20"].iloc[lo:t].values, BINS)
    up = (ret1.iloc[lo + 1:t + 1] > 0).values.astype(float)
    ok = ~np.isnan(ret1.iloc[lo + 1:t + 1].values).ravel()

    counts = np.zeros((len(BINS) + 1, len(BINS) + 1))
    ups = np.zeros_like(counts)
    np.add.at(counts, (b5.ravel()[ok], bd.ravel()[ok]), 1)
    np.add.at(ups, (b5.ravel()[ok], bd.ravel()[ok]), up.ravel()[ok])
    p = (ups + 1) / (counts + 2)

    today5 = np.digitize(panel["ret5"].iloc[t].values, BINS)
    todayd = np.digitize(panel["dist_ma20"].iloc[t].values, BINS)
    return pd.Series((p[today5, todayd] - 0.5) * 2, index=ret1.columns)


# Learned model: Ridge regression over all seven features, refit every day on the
# trailing year so it can only ever see the past. Ridge's L2 penalty keeps the
# coefficients small, which matters because these features are noisy and correlated.
# Training rows whose next-day move exceeded CLIP are dropped (news, not noise),
# and only the pattern the model actually learned is trusted at prediction time.
def ml(panel, t, ret1, window=250):
    lo = max(0, t - window)
    if t - lo < 30:
        return pd.Series(np.nan, index=ret1.columns)

    X = np.column_stack([panel[f].iloc[lo:t].values.ravel() for f in FEATURES])
    y = ret1.iloc[lo + 1:t + 1].values.ravel() * 100
    keep = ~(np.isnan(X).any(axis=1) | np.isnan(y)) & (np.abs(y) <= CLIP)

    model = Ridge(alpha=10.0).fit(X[keep], y[keep])

    x = np.column_stack([panel[f].iloc[t].values for f in FEATURES])
    pred = np.full(len(x), np.nan)
    ok = ~np.isnan(x).any(axis=1)
    pred[ok] = model.predict(x[ok])
    return pd.Series(pred, index=ret1.columns)
