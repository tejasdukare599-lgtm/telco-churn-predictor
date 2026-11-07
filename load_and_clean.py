import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB_PATH = "churn.db"
QUERIES_PATH = "sql/queries.sql"

EXPECTED = ["customerID", "tenure", "Contract", "InternetService", "PaymentMethod",
            "MonthlyCharges", "TotalCharges", "Churn"]


def clean(df):
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns {missing}. Your columns are: {list(df.columns)}")

    # TotalCharges is stored as text and has blank values for brand-new customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blanks = df["TotalCharges"].isna().sum()
    df = df.dropna(subset=["TotalCharges"]).copy()
    print(f"Dropped {blanks} rows with blank TotalCharges (customers with tenure 0)")

    # Turn the Yes/No target into 1/0 so models can use it.
    df["churn_flag"] = (df["Churn"] == "Yes").astype(int)

    # Make column names SQL-friendly (lowercase, no spaces).
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def run_queries(conn):
    path = Path(QUERIES_PATH)
    if not path.exists():
        print(f"\n(No {QUERIES_PATH} found, skipping queries.)")
        return
    for i, query in enumerate(q.strip() for q in path.read_text().split(";") if q.strip()):
        print(f"\n--- Query {i + 1} ---\n{query}\n")
        print(pd.read_sql(query, conn).to_string(index=False))


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python day1_load_and_clean.py data/Telco-Customer-Churn.csv")

    raw = pd.read_csv(sys.argv[1])
    print(f"Raw rows: {len(raw)}")
    print(f"Columns: {list(raw.columns)}")

    df = clean(raw)
    print(f"Clean rows: {len(df)}")
    print(f"Overall churn rate: {df['churn_flag'].mean():.1%}")

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    df.to_csv("data/clean_customers.csv", index=False)
    print(f"Saved table 'customers' to {DB_PATH} and data/clean_customers.csv")

    run_queries(conn)
    conn.close()


if __name__ == "__main__":
    main()
