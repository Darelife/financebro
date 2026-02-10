import pandas as pd
import re

def visible_len(s):
    """
    Input Structure:
    s (str): Input string containing ANSI codes.

    Who calls this function:
    Internal helper for test_prediction_accuracy.

    Task:
    Calculate the visible length of a string by stripping ANSI codes.

    Code Logic:
    1. Define regex for ANSI codes.
    2. Substitute ANSI codes with empty string.
    3. Return length of result.

    Output:
    int: Visible length of the string.
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', str(s)))

def print_table(data, headers):
    """
    Input Structure:
    data (list of dict): Row data.
    headers (list of str): Column headers.

    Who calls this function:
    test_prediction_accuracy.

    Task:
    Print a properly aligned table capable of handling ANSI color codes.

    Code Logic:
    1. Calculate maximum width for each column based on headers and data (using visible_len).
    2. Add a buffer to widths.
    3. Print header row using calculated widths.
    4. Print separator row.
    5. Iterate through data and print each row, padding values based on visible_len.

    Output:
    None.
    """
    if not data:
        return

    # Calculate column widths
    col_widths = {h: len(h) for h in headers}
    for row in data:
        for h in headers:
            val = str(row.get(h, ""))
            v_len = visible_len(val)
            if v_len > col_widths[h]:
                col_widths[h] = v_len

    # Add padding buffer
    for h in col_widths:
        col_widths[h] += 2

    # Print Header
    header_str = "".join(h.ljust(col_widths[h]) for h in headers)
    print(header_str)
    print("-" * len(header_str))

    # Print Rows
    for row in data:
        row_str = ""
        for h in headers:
            val = str(row.get(h, ""))
            v_len = visible_len(val)
            padding = col_widths[h] - v_len
            row_str += val + " " * padding
        print(row_str)

def test_prediction_accuracy(df):
    """
    Input Structure:
    df (pandas.DataFrame): The dataframe containing stock data with 'ltp' and 'prev_price'.

    Who calls this function:
    Main script when the -t or --test flag is provided.

    Task:
    Backtest the prediction logic by comparing "Yesterday's Prediction" against "Today's Actual" with detailed color-coded output in table format.

    Code Logic:
    1. Ensure numeric types for 'ltp' and 'prev_price'.
    2. Filter valid data.
    3. Iterate through recommended stocks (Top 5).
    4. Prepare "Yesterday's Recommendation" table data with Rank.
    5. Prepare "Today's Performance" table data with Rank and color coding.
    6. Print the tables using print_table helper.
    7. Calculate and print the investment summary with color coding.

    Output:
    pandas.DataFrame: The dataframe (unmodified in structure, but printed output is generated).
    """
    if df.empty:
        return df
    
    df['ltp'] = pd.to_numeric(df['ltp'], errors='coerce')
    
    if 'prev_price' not in df.columns:
        if 'previousClose' in df.columns:
             df['prev_price'] = df['previousClose']
        else:
            print("Error: 'prev_price' column not found for backtesting.")
            return df

    df['prev_price'] = pd.to_numeric(df['prev_price'], errors='coerce')
    
    df = df.dropna(subset=['ltp', 'prev_price'])

    recommendations = df.head(5).copy()

    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    # --- Yesterday's Recommendations Table ---
    print("\n----- These would have been the 5 stocks that I would have recommended -----")
    
    rec_data = []
    for idx, row in recommendations.iterrows():
        display_idx = recommendations.index.get_loc(idx) + 1
        rec_data.append({
            '#': str(display_idx),
            'Stock': row['symbol'],
            'Predicted Change': '+2.00%',
            "Yesterday's Closing": str(row['prev_price'])
        })
    
    print_table(rec_data, ['#', 'Stock', 'Predicted Change', "Yesterday's Closing"])
    
    # Predicted Investment Value
    predicted_pct = 2.0
    predicted_return = 100000 * (1 + predicted_pct / 100)
    profit = predicted_return - 100000
    print(f"\n{GREEN}If you invested 100,000 inr yesterday equally in these stocks, you would be predicted to have {int(predicted_return):,} inr. (+ {int(profit):,}){RESET}")

    # --- Today's Performance Table ---
    print("\n----- These are the best performing stocks for today -----")
    
    perf_data = []
    for idx, row in recommendations.iterrows():
        display_idx = recommendations.index.get_loc(idx) + 1
        
        # Calculate actual change percentage
        if row['prev_price'] != 0:
            actual_change_pct = ((row['ltp'] - row['prev_price']) / row['prev_price']) * 100
        else:
            actual_change_pct = 0.0

        if actual_change_pct >= 2.0:
            color = GREEN
        elif actual_change_pct >= 0:
            color = YELLOW
        else:
            color = RED
            
        sign = "+" if actual_change_pct >= 0 else ""
        change_str = f"{sign}{actual_change_pct:.2f}%"

        perf_data.append({
            '#': str(display_idx),
            'Stock': row['symbol'],
            'Actual Change': f"{color}{change_str}{RESET}",
            'Opening': f"{color}{row.get('open_price', 'N/A')}{RESET}",
            'Closing': f"{color}{row['ltp']}{RESET}",
            'Max': f"{color}{row.get('high_price', 'N/A')}{RESET}",
            'Min': f"{color}{row.get('low_price', 'N/A')}{RESET}"
        })

    print_table(perf_data, ['#', 'Stock', 'Actual Change', 'Opening', 'Closing', 'Max', 'Min'])

    # --- Investment Summary ---
    investment_per_stock = 20000
    total_investment = 100000
    
    recommendations['Units'] = investment_per_stock / recommendations['prev_price']
    recommendations['Current_Value'] = recommendations['Units'] * recommendations['ltp']
    
    total_current_value = recommendations['Current_Value'].sum()
    profit_loss = total_current_value - total_investment
    
    sign = "+" if profit_loss >= 0 else "-"
    pl_color = GREEN if profit_loss >= 0 else RED
    
    print(f"\n{pl_color}If you invested {total_investment:,} inr yesterday equally in the stocks i recommended, You would have had {int(total_current_value):,} inr. ({sign} {abs(int(profit_loss)):,}){RESET}")
    
    return df

