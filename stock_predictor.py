import argparse
import sys
from utils import data_fetcher, predictor, historical, backtester

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
    
    parser.add_argument('-p', '--predict', action='store_true', help='Make a prediction about top 5 stocks')
    parser.add_argument('-d', '--date', type=str, metavar='ddmmyyyy', help='Use data upto a specific date')
    parser.add_argument('-t', '--test', action='store_true', help='Test accuracy by comparing yesterday\'s data with today')
    
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
        
        if 'Predicted_Value' in df.columns:
            columns_to_show = ['symbol', 'ltp', 'Predicted_Value']
            print("\n----- Top 5 Predicted Future Values -----")
            df_sorted = df.sort_values(by='Predicted_Value', ascending=False)
            print(df_sorted[columns_to_show].head(5).to_string(index=False))
            
            print("\n----- All Stocks with Predictions -----")
            print(df[['symbol', 'ltp', 'Predicted_Value']].to_string(index=False))

    if args.test:
        print("\nRunning Backtest Mode...")
        backtester.test_prediction_accuracy(df)

    if args.date and not args.predict and not args.test:
        print(f"\nData for {args.date} (Simulated/Proxy):")
        print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
