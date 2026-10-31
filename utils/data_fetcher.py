import pandas as pd
import yfinance as yf
from nsetools import Nse


def get_candles(universe):
    """
    Input Structure:
    universe (tuple): NSE symbols to download.

    Who calls this function:
    The -p path in the main script.

    Task:
    Download two years of daily candles for the universe.

    Code Logic:
    1. Download daily OHLCV data from yfinance with the .NS suffix appended.
    2. Split it into closing-price and volume DataFrames (dates x symbols).
    3. Return both.

    Output:
    tuple: (close, volume) DataFrames, or (None, None) if the download fails.
    """
    try:
        data = yf.download(" ".join(s + ".NS" for s in universe), period="2y",
                           group_by="ticker", auto_adjust=True, progress=False)
        if data.empty:
            return None, None
        close = pd.DataFrame({s: data[s + ".NS"]["Close"] for s in universe})
        volume = pd.DataFrame({s: data[s + ".NS"]["Volume"] for s in universe})
        return close, volume
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None


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
