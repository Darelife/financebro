import pandas as pd

def predict_future_values(df):
    """
    Input Structure:
    df (pandas.DataFrame): A DataFrame containing current stock market data, including 'ltp' (Last Traded Price).

    Who calls this function:
    Main script when the -p or --predict flag is used.

    Task:
    Apply a prediction heuristic to estimate future stock values.

    Code Logic:
    1. Check if the DataFrame is empty.
    2. Define a simple prediction heuristic (e.g., a flat 2% increase for demonstration).
    3. Calculate 'Predicted_Value' = 'ltp' * 1.02.
    4. Return the DataFrame with the new prediction column.

    Output:
    pandas.DataFrame: The input DataFrame enriched with a 'Predicted_Value' column.
    """
    if df.empty:
        return df
    
    df['Predicted_Value'] = df['ltp'] * 1.02
    
    return df
