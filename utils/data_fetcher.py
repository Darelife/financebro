import pandas as pd
from nsetools import Nse


def get_market_data():
    """
    Input Structure:
    None.

    Who calls this function:
    Main script or other utility functions that require raw market data.

    Task:
    Fetch the list of top gaining stocks from the NSE and prepare it for analysis.

    Code Logic:
    1. Initialize the Nse object.
    2. Call get_top_gainers() to retrieve market data.
    3. Convert the returned data into a pandas DataFrame.
    4. Convert relevant numeric columns to proper numeric types.
    5. Sort the DataFrame based on percentage change.
    6. Reset index for clean ordering.
    7. Return the processed DataFrame.

    Output:
    pandas.DataFrame: A cleaned and sorted DataFrame containing top gainers data. Returns an empty DataFrame if fetch fails.
    """
    try:
        nse = Nse()
        top_gainers = nse.get_top_gainers("NIFTY")

        if not top_gainers:
            return pd.DataFrame()

        df = pd.DataFrame(top_gainers)

        numeric_cols = [
            "open_price",
            "high_price",
            "low_price",
            "ltp",
            "prev_price",
            "net_price",
            "turnover",
            "perChange",
            "pChange",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "pChange" in df.columns:
            df = df.sort_values(by="pChange", ascending=False)
        elif "perChange" in df.columns:
            df = df.sort_values(by="perChange", ascending=False)

        df = df.reset_index(drop=True)

        print(df)
        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
