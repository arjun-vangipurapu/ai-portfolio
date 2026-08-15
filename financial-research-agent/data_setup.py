# data_setup.py
import yfinance as yf
import sqlite3
import os
import pandas as pd

os.makedirs("data", exist_ok=True)

def setup_database():
    print("Setting up SQLite database...")
    conn = sqlite3.connect("data/metrics.db")
    
    # clear existing data to avoid duplicates on re-run
    cursor = conn.cursor()
    for table in ["stock_history", "quarterly_financials", "company_metrics"]:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()

    tickers = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "INFY": "Infosys"
    }

    for symbol, name in tickers.items():
        print(f"Downloading {name} ({symbol})...")
        ticker = yf.Ticker(symbol)

        # Stock price history — 2 years
        hist = ticker.history(period="2y")
        hist["symbol"] = symbol
        hist["company"] = name
        hist.to_sql("stock_history", conn,
                   if_exists="append", index=True)
        print(f"  ✅ Stock history: {len(hist)} rows")

        # Quarterly financials
        # Quarterly financials
        try:
            financials = ticker.quarterly_income_stmt
            if financials is not None and not financials.empty:
                # transpose and reset
                df = financials.T.reset_index()
                df.columns = [str(c) for c in df.columns]  # fix column names
                df["symbol"] = symbol
                df["company"] = name
                # drop columns with all nulls
                df = df.dropna(axis=1, how="all")
                df.to_sql("quarterly_financials", conn,
                        if_exists="append", index=False)
                print(f"  ✅ Quarterly financials: {len(df)} rows")
        except Exception as e:
            print(f"  ⚠️ Financials error: {e}")

        # Key metrics
        try:
            info = ticker.info
            metrics = {
                "symbol": symbol,
                "company": name,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margins": info.get("grossMargins"),
                "operating_margins": info.get("operatingMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
            }
            pd.DataFrame([metrics]).to_sql("company_metrics", conn,
                                          if_exists="append", index=False)
            print(f"  ✅ Key metrics saved")
        except Exception as e:
            print(f"  ⚠️ Metrics error: {e}")

    print("\nAdding revenue summary table...")
    revenue_data = [
        # Apple quarterly revenue (from earnings calls)
        ("AAPL", "Apple", "Q4 2024", 94.9, 6.0),
        ("AAPL", "Apple", "Q3 2024", 85.8, 5.0),
        ("AAPL", "Apple", "Q2 2024", 90.8, 4.0),
        ("AAPL", "Apple", "Q1 2024", 119.6, 2.0),
        # Microsoft quarterly revenue
        ("MSFT", "Microsoft", "Q4 2024", 64.7, 15.0),
        ("MSFT", "Microsoft", "Q3 2024", 61.9, 17.0),
        ("MSFT", "Microsoft", "Q2 2024", 62.0, 18.0),
        ("MSFT", "Microsoft", "Q1 2024", 56.5, 13.0),
        # Infosys quarterly revenue
        ("INFY", "Infosys", "Q1 2027", 5.08, 2.88),
        ("INFY", "Infosys", "Q4 2026", 4.94, 2.0),
        ("INFY", "Infosys", "Q3 2026", 4.89, 3.0),
        ("INFY", "Infosys", "Q2 2026", 4.71, 1.5),
    ]

    revenue_df = pd.DataFrame(revenue_data, columns=[
        "symbol", "company", "quarter",
        "revenue_billion", "yoy_growth_pct"
    ])
    revenue_df.to_sql("revenue_summary", conn,
                    if_exists="replace", index=False)
    print(f"  ✅ Revenue summary: {len(revenue_df)} rows")

    conn.close()
    print("\n✅ Database setup complete: data/metrics.db")

if __name__ == "__main__":
    setup_database()