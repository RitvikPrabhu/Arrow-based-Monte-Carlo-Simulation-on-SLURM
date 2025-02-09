#!/usr/bin/env python3
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def simulate_faang(
    ticker, cal_start="2018-01-01", cal_end="2018-12-31", sim_days=30, num_paths=10000
):
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
    return predicted_mean, actual_price, error


def main():
    faang = ["AAPL", "AMZN", "NFLX", "GOOGL", "META"]
    results = []
    start_time = time.time()
    for ticker in faang:
        pred, actual, err = simulate_faang(ticker)
        results.append(
            {
                "ticker": ticker,
                "predicted_mean": pred,
                "actual_price": actual,
                "error": err,
            }
        )
        print(f"{ticker}: Predicted={pred:.2f}, Actual={actual:.2f}, Error={err:.2f}")
    total_time = time.time() - start_time
    print(f"Baseline total time: {total_time:.2f} seconds")
    df = pd.DataFrame(results)
    df["abs_error"] = df["error"].abs()
    ax = df.set_index("ticker")["abs_error"].plot(
        kind="bar", title="Baseline: Absolute Error per Ticker"
    )
    ax.set_ylabel("Absolute Error")
    plt.savefig("baseline_errors.png")
    plt.show()


if __name__ == "__main__":
    main()
