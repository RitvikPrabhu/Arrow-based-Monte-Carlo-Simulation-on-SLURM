#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.flight as flight
import yfinance as yf


def simulate_faang(
    ticker, cal_start="2018-01-01", cal_end="2018-12-31", sim_days=30, num_paths=10000
):
    start_time = time.time()
    data = yf.download(ticker, start=cal_start, end=cal_end)
    prices = data["Adj Close"].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_mean = log_returns.mean()
    daily_vol = log_returns.std()
    S0 = prices.iloc[-1]
    sims = np.empty((num_paths, sim_days))
    sims[:, 0] = S0 * np.exp(
        (daily_mean - 0.5 * daily_vol**2) + daily_vol * np.random.randn(num_paths)
    )
    for t in range(1, sim_days):
        sims[:, t] = sims[:, t - 1] * np.exp(
            (daily_mean - 0.5 * daily_vol**2) + daily_vol * np.random.randn(num_paths)
        )
    predicted_mean = np.mean(sims[:, -1])
    sim_start_date = prices.index[-1] + pd.Timedelta(days=1)
    sim_end_date = prices.index[-1] + pd.Timedelta(days=sim_days)
    actual_data = yf.download(
        ticker,
        start=sim_start_date.strftime("%Y-%m-%d"),
        end=sim_end_date.strftime("%Y-%m-%d"),
    )
    actual_price = (
        actual_data["Adj Close"].iloc[-1] if not actual_data.empty else float("nan")
    )
    error = predicted_mean - actual_price
    runtime = time.time() - start_time
    result = {
        "ticker": [ticker],
        "cal_start": [cal_start],
        "cal_end": [cal_end],
        "last_cal_date": [prices.index[-1].strftime("%Y-%m-%d")],
        "sim_start": [sim_start_date.strftime("%Y-%m-%d")],
        "sim_end": [sim_end_date.strftime("%Y-%m-%d")],
        "predicted_mean": [predicted_mean],
        "actual_price": [actual_price],
        "error": [error],
        "daily_mean": [daily_mean],
        "daily_vol": [daily_vol],
        "runtime": [runtime],
    }
    return pa.Table.from_pydict(result)


def main():
    faang = ["AAPL", "AMZN", "NFLX", "GOOGL", "META"]
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is not None:
        idx = (int(task) - 1) % len(faang)
        ticker = faang[idx]
    else:
        ticker = faang[0]
    if len(sys.argv) < 2:
        print("Usage: python worker.py <flight_endpoint>")
        sys.exit(1)
    flight_endpoint = sys.argv[1]
    print(f"Worker for {ticker} connecting to {flight_endpoint}")
    try:
        client = flight.FlightClient(flight_endpoint)
    except Exception as e:
        print(f"Error connecting: {e}")
        sys.exit(1)
    table = simulate_faang(ticker)
    descriptor = flight.FlightDescriptor.for_path("faang_simulation")
    try:
        writer, _ = client.do_put(descriptor)
        writer.write_table(table)
        writer.close()
        print("Worker: Results sent.")
    except Exception as e:
        print(f"Error sending: {e}")


if __name__ == "__main__":
    main()
