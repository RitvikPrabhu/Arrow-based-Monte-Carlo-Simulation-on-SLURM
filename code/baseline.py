import time
import argparse
import logging
import numpy as np
import pandas as pd
import yfinance as yf
# import matplotlib.pyplot as plt

def simulate_ticker(ticker, cal_start="2012-01-01", cal_end="2022-12-31", sim_days=365, num_paths=1000000):
    logging.info(f"Starting baseline simulation for {ticker}.")
    t0 = time.time()
    data = yf.download(ticker, start=cal_start, end=cal_end, progress=False)
    if data.empty:
        logging.error(f"No historical data for {ticker} during calibration period {cal_start} to {cal_end}.")
        return None, None, None, None
    logging.info("Historical data downloaded.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    prices = data["Close"].dropna()
    if prices.empty:
        logging.error(f"No price data available for {ticker}.")
        return None, None, None, None
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_mean = log_returns.mean()
    daily_vol = log_returns.std()
    S0 = prices.iloc[-1]
    sims = np.empty((num_paths, sim_days))
    sims[:, 0] = S0 * np.exp((daily_mean - 0.5*daily_vol**2) + daily_vol*np.random.randn(num_paths))
    for t in range(1, sim_days):
        sims[:, t] = sims[:, t-1] * np.exp((daily_mean - 0.5*daily_vol**2) + daily_vol*np.random.randn(num_paths))
    predicted_mean = np.mean(sims[:, -1])
    sim_start_date = prices.index[-1] + pd.Timedelta(days=1)
    sim_end_date = prices.index[-1] + pd.Timedelta(days=sim_days)
    actual_data = yf.download(ticker, start=sim_start_date.strftime("%Y-%m-%d"), end=sim_end_date.strftime("%Y-%m-%d"), progress=False)
    if not actual_data.empty:
        if "Adj Close" in actual_data.columns:
            actual_price = actual_data["Adj Close"].iloc[-1]
        else:
            actual_price = actual_data["Close"].iloc[-1]
    else:
        actual_price = float("nan")
    runtime = time.time() - t0
    logging.info(f"Baseline simulation for {ticker} complete in {runtime:.2f} seconds.")
    error = predicted_mean - actual_price
    return predicted_mean, actual_price, error, runtime

def main():
    parser = argparse.ArgumentParser(description="Baseline simulation for FAANG tickers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--sim-days", type=int, default=180, help="Number of simulation days (default 180)")
    parser.add_argument("--num-paths", type=int, default=30000, help="Number of simulation paths per ticker (default 30000)")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
    
    tickers = ["AAPL", "AMZN", "NFLX", "GOOGL", "META"]
    results = []
    
    for ticker in tickers:
        logging.info(f"Simulating for ticker {ticker}")
        pred, actual, err, runtime = simulate_ticker(ticker, sim_days=args.sim_days, num_paths=args.num_paths)
        if pred is None:
            logging.error(f"Skipping {ticker} due to errors.")
            continue
        results.append({
            "ticker": ticker,
            "predicted_mean": pred,
            "actual_price": actual,
            "error": err,
            "runtime": runtime
        })
        logging.info(f"{ticker}: Predicted = {pred:.2f}, Actual = {actual.iloc[0]:.2f}, Error = {err.iloc[0]:.2f}, Runtime = {runtime:.2f} sec")

    df = pd.DataFrame(results)
    df.to_csv("results/baseline_results.csv", index=False)
    logging.info("Saved baseline results to baseline_results.csv")
    
    # fig, ax = plt.subplots(figsize=(8,6))
    # x = range(len(df))
    # width = 0.35
    # ax.bar([xi - width/2 for xi in x], df['predicted_mean'], width, label="Predicted", color="green")
    # ax.bar([xi + width/2 for xi in x], df['actual_price'], width, label="Actual", color="red")
    # ax.set_xticks(x)
    # ax.set_xticklabels(df['ticker'])
    # ax.set_ylabel("Price")
    # ax.set_title("Baseline Simulation: Predicted vs Actual Prices for FAANG")
    # ax.legend()
    # plt.savefig("baseline_combined.png")
    # plt.show()

if __name__ == "__main__":
    main()
