import argparse
import datetime
import json
import os
import sys

from utils import backtester, data_fetcher, historical, predictor, simulator


def parse_arguments():
    """
    Input Structure:
    None (uses sys.argv).

    Who calls this function:
    Main execution block.

    Task:
    Parse command line arguments for controlling prediction and testing behavior.

    Code Logic:
    1. Define arguments for prediction, date filtering, testing, and simulation.
    2. Parse command line arguments.
    3. If no arguments are provided, display help and exit.
    4. Return parsed arguments.

    Output:
    argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Indian Stock Market Predictor")

    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Make a prediction about top 5 stocks",
    )
    parser.add_argument(
        "-d",
        "--date",
        type=str,
        metavar="ddmmyyyy",
        help="Use data upto a specific date",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test accuracy by comparing yesterday's data with today",
    )
    parser.add_argument(
        "-s",
        "--simulate",
        type=int,
        metavar="DAYS",
        help="Simulate the strategy over the last DAYS trading days",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def main():
    """
    Input Structure:
    None.

    Who calls this function:
    Script entry point.

    Task:
    Control the execution flow for fetching data, predicting values, and backtesting.

    Code Logic:
    1. Parse command line arguments.
    2. Run simulation and exit if requested.
    3. Fetch market data.
    4. Exit if no data is available.
    5. Apply date filtering if specified.
    6. Run prediction logic if requested.
    7. Sort and display top predicted stocks.
    8. Store top predictions in a JSON file.
    9. Run backtesting if requested.
    10. Display filtered data if only date is provided.

    Output:
    None (prints results and stores predictions in a JSON file).
    """
    args = parse_arguments()

    if args.simulate:
        simulator.simulate(args.simulate)
        return

    print("Fetching market data...")
    df = data_fetcher.get_market_data()

    if df.empty:
        print("No data received. Exiting.")
        return

    if args.date:
        df = historical.filter_by_date(df, args.date)

    if args.predict:
        print("\nRunning Prediction Mode...")
        df = predictor.predict_future_values(df)

        if "Predicted_Return" in df.columns:
            columns_to_show = ["symbol", "ltp", "Predicted_Return"]
            df_sorted = df.sort_values(by="Predicted_Return", ascending=False)

            print("\n----- Top 5 Predicted Future Values -----")
            print(df_sorted[columns_to_show].head(5).to_string(index=False))

            data = {}
            for _, row in df_sorted.head(5).iterrows():
                data[row["symbol"]] = {
                    "Predicted_Return": row["Predicted_Return"],
                    "ltp": row["ltp"],
                    "high_price": row["high_price"],
                    "low_price": row["low_price"],
                    "prev_price": row["prev_price"],
                    "net_price": row["net_price"],
                    "turnover": row["turnover"],
                    "open_price": row["open_price"],
                }

            date = datetime.datetime.now().strftime("%d-%m-%Y")
            filename = "predictionData.json"

            if os.path.exists(filename):
                with open(filename, "r") as f:
                    try:
                        old = json.load(f)
                    except json.JSONDecodeError:
                        old = {}
            else:
                old = {}

            old[date] = data

            with open(filename, "w") as f:
                json.dump(old, f, indent=4)

            print("\n----- All Stocks with Predictions -----")
            print(df[["symbol", "ltp", "Predicted_Return"]].to_string(index=False))

    if args.test:
        print("\nRunning Backtest Mode...")
        backtester.test_prediction_accuracy(df)

    if args.date and not args.predict and not args.test:
        print(f"\nData for {args.date} (Simulated/Proxy):")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
