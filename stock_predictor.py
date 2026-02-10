from nsetools import Nse
import pandas as pd

def main():
    """
    Fetches the top gainers from the NSE and prints them in a formatted table.
    """
    try:
        nse = Nse()
        top_gainers = nse.get_top_gainers()
        
        if top_gainers:
            df = pd.DataFrame(top_gainers)
            
            columns = ['symbol', 'ltp', 'perChange', 'open_price', 'high_price', 'low_price']
            
            df['perChange'] = pd.to_numeric(df['perChange'], errors='coerce')
            df = df.sort_values(by='perChange', ascending=False)
            
            print("\n----- Top Gainers (Highest Return Today) -----")
            print(df[columns].head(10).to_string(index=False))
        else:
            print("No data available.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
