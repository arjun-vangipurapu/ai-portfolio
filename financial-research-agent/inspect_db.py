# inspect_db.py
import sqlite3

conn = sqlite3.connect("data/metrics.db")
cursor = conn.cursor()

# list all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", tables)

# count rows per table
for table in tables:
    name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {name}")
    print(f"  {name}: {cursor.fetchone()[0]} rows")

# preview company_metrics
print("\nCompany Metrics:")
cursor.execute("SELECT symbol, company, market_cap, pe_ratio, revenue_growth FROM company_metrics")
for row in cursor.fetchall():
    print(f"  {row}")

conn.close()