from nsetools import Nse
import pandas as pd

def get_market_data():
    """
    Input Structure:
    None.

    Who calls this function:
    Main script or other utility functions that need raw market data.

    Task:
    Fetch the list of top gainers from the National Stock Exchange (NSE) using nsetools.

    Code Logic:
    1. Initialize the Nse object.
    2. Call get_top_gainers() to retrieve data.
    3. If data is returned, convert it to a pandas DataFrame.
    4. Ensure 'pChange' or 'perChange' column is numeric for sorting.
    5. Sort the DataFrame by percentage change in descending order.
    6. Return the sorted DataFrame.

    Output:
    pandas.DataFrame: A DataFrame containing top gainers data, sorted by percentage change. Returns an empty DataFrame if fetch fails.
    """
    try:
        nse = Nse()
        top_gainers = nse.get_top_gainers()
        
        if top_gainers:
            df = pd.DataFrame(top_gainers)
            
            if 'pChange' in df.columns:
                sort_col = 'pChange'
            elif 'perChange' in df.columns:
                sort_col = 'perChange'
            else:
                sort_col = None

            if sort_col:
                df[sort_col] = pd.to_numeric(df[sort_col], errors='coerce')
                df = df.sort_values(by=sort_col, ascending=False)
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
