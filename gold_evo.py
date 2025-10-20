# File: gold_semester_evolution.py
# Fetch and plot gold price evolution for the current semester (Jan-Jun or Jul-Dec).
# Requires: yfinance, pandas, matplotlib
# Install: pip install yfinance pandas matplotlib

import sys
from datetime import date, timedelta
import argparse

try:
    import yfinance as yf
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception:
    sys.stderr.write("Missing dependency: please pip install yfinance pandas matplotlib\n")
    sys.exit(1)


def semester_start(today: date) -> date:
    return date(today.year, 1, 1) if today.month <= 6 else date(today.year, 7, 1)


def fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                     progress=False, auto_adjust=True)  # auto_adjust True pour éviter warning
    if df.empty:
        raise RuntimeError(f"No data returned for ticker {ticker} between {start} and {end}.")
    # ensure timezone-naive index for plotting/annotations
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)
    return df


def summary(df: pd.DataFrame) -> dict:
    # Sélection du scalar, même si c'est un Series
    o = float(df["Open"].iloc[0])
    c = float(df["Close"].iloc[-1])
    pct = (c - o) / o * 100

    # Gestion scalar/multi-colonnes pour High/Low
    if isinstance(df["High"], pd.DataFrame):
        high_val = float(df["High"].iloc[:, 0].max())
        low_val = float(df["Low"].iloc[:, 0].min())
        hi_idx = df["High"].iloc[:, 0].idxmax()
        lo_idx = df["Low"].iloc[:, 0].idxmin()
    else:
        high_val = float(df["High"].max())
        low_val = float(df["Low"].min())
        hi_idx = df["High"].idxmax()
        lo_idx = df["Low"].idxmin()

    high_date = hi_idx.strftime('%Y-%m-%d') if isinstance(hi_idx, pd.Timestamp) else str(hi_idx)
    low_date = lo_idx.strftime('%Y-%m-%d') if isinstance(lo_idx, pd.Timestamp) else str(lo_idx)

    daily_ret = df["Close"].pct_change().dropna()
    mean_ret = float(daily_ret.mean() * 100)
    std_ret = float(daily_ret.std() * 100)
    best_day = float(daily_ret.max() * 100)
    worst_day = float(daily_ret.min() * 100)

    return {
        "start_open": o,
        "end_close": c,
        "pct_change": pct,
        "high": high_val,
        "high_date": high_date,
        "low": low_val,
        "low_date": low_date,
        "mean_daily_return_pct": mean_ret,
        "std_daily_return_pct": std_ret,
        "best_day_pct": best_day,
        "worst_day_pct": worst_day,
    }



def plot_prices(df: pd.DataFrame, out_file: str, title: str):
    # Vérifie si le style existe, sinon style par défaut
    if "seaborn-darkgrid" in plt.style.available:
        plt.style.use("seaborn-darkgrid")
    else:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Close"], label="Close", color="tab:blue")

    # 20-day moving average
    ma20 = df["Close"].rolling(window=20, min_periods=1).mean()
    ax.plot(df.index, ma20, label="MA20", color="tab:orange", linewidth=1.5)

    # Mark high and low
    if isinstance(df["High"], pd.DataFrame):
        hi_idx = df["High"].iloc[:, 0].idxmax()
        lo_idx = df["Low"].iloc[:, 0].idxmin()
        high_val = float(df["High"].iloc[:, 0].max())
        low_val = float(df["Low"].iloc[:, 0].min())
    else:
        hi_idx = df["High"].idxmax()
        lo_idx = df["Low"].idxmin()
        high_val = float(df["High"].max())
        low_val = float(df["Low"].min())

    ax.scatter([hi_idx], [high_val], color="green", marker="^", zorder=5)
    ax.scatter([lo_idx], [low_val], color="red", marker="v", zorder=5)
    ax.annotate(f"High {high_val:.2f}", xy=(hi_idx, high_val), xytext=(10, 10),
                textcoords="offset points", fontsize=8)
    ax.annotate(f"Low {low_val:.2f}", xy=(lo_idx, low_val), xytext=(10, -15),
                textcoords="offset points", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Gold price evolution this semester")
    parser.add_argument("--ticker", default="GC=F", help="Ticker to use (default: GC=F for gold futures). Try GLD for ETF.")
    parser.add_argument("--out", default="gold_semester.png", help="Output plot file")
    parser.add_argument("--csv", default="gold_semester.csv", help="Save fetched data to CSV")
    args = parser.parse_args()

    today = date.today()
    start = semester_start(today)

    try:
        df = fetch_prices(args.ticker, start, today)
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)

    s = summary(df)
    title = f"{args.ticker} price since {start.isoformat()} (as of {today.isoformat()})"

    plot_prices(df, args.out, title)

    # save CSV
    df.to_csv(args.csv)

    # textual report
    print("------------------ Semester Summary ----------------\n")
    print(f"Period start (open): {float(s['start_open']):.2f} USD")
    print("\n-------------------------------------------------------------------\n")
    print(f"Period end (close):  {float(s['end_close']):.2f} USD")
    print("\n-------------------------------------------------------------------\n")
    print(f"Change over semester: {float(s['pct_change']):.2f}%")
    print("\n-------------------------------------------------------------------\n")
    print(f"High: {float(s['high']):.2f} USD on {s['high_date']}")
    print("\n-------------------------------------------------------------------\n")
    print(f"Low:  {float(s['low']):.2f} USD on {s['low_date']}")   
    print("\n-------------------------------------------------------------------\n")
    print(f"Mean daily return: {float(s['mean_daily_return_pct']):.3f}%")
    print("\n-------------------------------------------------------------------\n")
    print(f"Std dev daily return: {float(s['std_daily_return_pct']):.3f}%")
    print("\n-------------------------------------------------------------------\n")
    print(f"Best single day: {float(s['best_day_pct']):.2f}%")
    print("\n-------------------------------------------------------------------\n")
    print(f"Worst single day: {float(s['worst_day_pct']):.2f}%")



if __name__ == "__main__":
    main()
