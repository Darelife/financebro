import pandas as pd

def test_prediction_accuracy(df):
    """
    Input Structure:
    df (pandas.DataFrame): The dataframe containing stock data with 'ltp' and 'prev_price'.

    Who calls this function:
    Main script when the -t or --test flag is provided.

    Task:
    Backtest the prediction logic by comparing "Yesterday's Prediction" against "Today's Actual".

    Code Logic:
    1. Identify 'Yesterday's Close' as 'prev_price' (Previous Close).
    2. Calculate 'Predicted_For_Today' using the same logic as the predictor module (e.g., prev_price * 1.02).
    3. Compare 'Predicted_For_Today' with 'ltp' (Last Traded Price, which is Today's Actual).
    4. Calculate the difference and accuracy percentage.
    5. Add these as new columns to the DataFrame.
    6. Return the DataFrame with comparison results.

    Output:
    pandas.DataFrame: The dataframe enriched with backtesting metrics.
    """
    if df.empty:
        return df
    
    if 'prev_price' not in df.columns:
        print("Error: 'prev_price' column not found for backtesting.")
        return df

    df['Yesterday_Close'] = pd.to_numeric(df['prev_price'], errors='coerce')
    
    df['Predicted_Today'] = df['Yesterday_Close'] * 1.02
    
    df['Actual_Today'] = df['ltp']
    
    df['Error'] = df['Predicted_Today'] - df['Actual_Today']
    df['Accuracy_%'] = 100 - (abs(df['Error']) / df['Actual_Today'] * 100)
    
    return df
