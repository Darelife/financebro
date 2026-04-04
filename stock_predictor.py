import argparse
import datetime
import json
import sys

from utils import backtester, data_fetcher, historical, predictor


def parse_arguments():
    """
    Input Structure:
    None (uses sys.argv).

    Who calls this function:
    Main execution block.

    Task:
    Parse command line arguments for the stock predictor.

    Code Logic:
    1. Define arguments: -p/--predict, -d/--date, -t/--test.
    2. Parse arguments.
    3. If no arguments are provided, print help and exit.

    Output:
    argparse.Namespace: The parsed arguments.
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
    Orchestrate the stock prediction/testing flow based on user flags.

    Code Logic:
    1. Parse arguments.
    2. Fetch market data (Top Gainers).
    3. If no data, exit.
    4. Apply date filtering if -d is present.
    5. Apply prediction logic if -p is present.
    6. Apply backtesting logic if -t is present.
    7. If only date is present, show data.

    Output:
    None (Prints to stdout).
    """
    args = parse_arguments()

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

        print(df.columns)
        if "Predicted_Value" in df.columns:
            columns_to_show = ["symbol", "ltp", "Predicted_Value"]
            print("\n----- Top 5 Predicted Future Values -----")
            df_sorted = df.sort_values(by="Predicted_Value", ascending=False)
            print(df_sorted[columns_to_show].head(5).to_string(index=False))

            data = {}
            count = 0
            for i, row in df_sorted.iterrows():
                # if i >= 5:
                #     break
                count += 1
                # data[row["symbol"]] = row["Predicted_Value"]
                # Index(['symbol', 'series', 'open_price', 'high_price', 'low_price', 'ltp',
                #        'prev_price', 'net_price', 'trade_quantity', 'turnover', 'market_type',
                #        'ca_ex_dt', 'ca_purpose', 'perChange', 'Predicted_Value'],
                #       dtype='object')
                entry = {
                    "Predicted_Value": row["Predicted_Value"],
                    "ltp": row["ltp"],
                    "high_price": row["high_price"],
                    "low_price": row["low_price"],
                    "prev_price": row["prev_price"],
                    "net_price": row["net_price"],
                    "turnover": row["turnover"],
                    "open_price": row["open_price"],
                }
                data[row["symbol"]] = entry
                if count >= 5:
                    break

            date = datetime.datetime.now().strftime("%d-%m-%Y")
            # data = {datetime.datetime.now().strftime("%d-%m-%Y"): data}
            with open("predictionData.json", "r") as f:
                old = json.load(f)

            with open("predictionData.json", "w") as f:
                old[date] = data
                json.dump(old, f, indent=4)

            print("\n----- All Stocks with Predictions -----")
            print(df[["symbol", "ltp", "Predicted_Value"]].to_string(index=False))

    if args.test:
        print("\nRunning Backtest Mode...")
        backtester.test_prediction_accuracy(df)

    if args.date and not args.predict and not args.test:
        print(f"\nData for {args.date} (Simulated/Proxy):")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
