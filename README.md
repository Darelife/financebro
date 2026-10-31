# financeBro

Predicts the top 5 NSE stocks most likely to keep rising tomorrow, then grades itself on how the previous picks actually performed.

Ranks every NIFTY 50 stock with an ensemble of four strategies — momentum, mean reversion, an empirical next-day-up probability, and a Ridge model trained on the trailing year (next-day moves beyond 3% are treated as news and dropped from training).

## Setup

```bash
pip install nsetools pandas yfinance scikit-learn matplotlib
```

## Run

```bash
# Predict today's top 5 stocks (saves them to predictionData.json)
python stock_predictor.py -p

# Backtest: compare the last saved prediction against today's actual prices
python stock_predictor.py -t

# Simulate the strategy over the last N trading days of NIFTY 50 history
# (also saves a per-strategy comparison chart to strategy_comparison.png)
python stock_predictor.py -s 120
```

Optional: `-d ddmmyyyy` filters data up to a specific date, e.g. `python stock_predictor.py -d 01012026`.

## Daily loop

1. Run `python stock_predictor.py -p` during or after market hours.
2. The next day, run `python stock_predictor.py -t` — it shows how the previous picks did and saves their real returns.
3. Repeat. The saved returns feed back into the predictor, so predictions get calibrated over time.
