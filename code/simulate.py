import time
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import yfinance as yf

def robust_download(ticker, start, end, retries=3, delay=5):
    for attempt in range(1, retries+1):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                logging.info(f"Download for {ticker} successful on attempt {attempt}.")
                return data
            else:
                logging.warning(f"Attempt {attempt}: empty data for {ticker}.")
        except Exception as e:
            logging.error(f"Attempt {attempt}: Error downloading {ticker}: {e}")
        if attempt < retries:
            logging.info(f"Sleeping {delay} seconds before retry...")
            time.sleep(delay)
    logging.error(f"Failed to download data for {ticker} after {retries} attempts.")
    return pd.DataFrame()  # empty

def simulate_ticker(ticker, cal_start="2022-01-01", cal_end="2022-12-31",
                    sim_days=180, num_paths=1000000):

    logging.info(f"Starting simulation for {ticker} with {num_paths} paths.")
    t0 = time.time()

    data = robust_download(ticker, cal_start, cal_end)
    if data.empty:
        logging.error(f"No data for {ticker}. Returning dummy partial result.")
        return pa.Table.from_pydict({
            "ticker": [ticker],
            "partial_mean": [float("nan")],
            "num_paths": [0],
            "actual_price": [float("nan")],
            "runtime": [0.0]
        })

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        logging.error(f"Data for {ticker} lacks 'Close' column. Returning dummy.")
        return pa.Table.from_pydict({
            "ticker": [ticker],
            "partial_mean": [float("nan")],
            "num_paths": [0],
            "actual_price": [float("nan")],
            "runtime": [0.0]
        })

    prices = data["Close"].dropna()
    if prices.empty:
        logging.error(f"No 'Close' data for {ticker} in that period. Returning dummy.")
        return pa.Table.from_pydict({
            "ticker": [ticker],
            "partial_mean": [float("nan")],
            "num_paths": [0],
            "actual_price": [float("nan")],
            "runtime": [0.0]
        })

    try:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        daily_mean = float(log_returns.mean())
        daily_vol = float(log_returns.std())
        S0 = float(prices.iloc[-1])
        sims = np.empty((num_paths, sim_days), dtype=np.float64)
        sims[:, 0] = S0 * np.exp((daily_mean - 0.5*daily_vol**2) + daily_vol*np.random.randn(num_paths))
        for t in range(1, sim_days):
            sims[:, t] = sims[:, t-1] * np.exp((daily_mean - 0.5*daily_vol**2) + daily_vol*np.random.randn(num_paths))
        part_mean = np.mean(sims[:, -1])
    except Exception as e:
        logging.error(f"Error during simulation for {ticker}: {e}")
        return pa.Table.from_pydict({
            "ticker": [ticker],
            "partial_mean": [float("nan")],
            "num_paths": [0],
            "actual_price": [float("nan")],
            "runtime": [time.time() - t0]
        })

    sim_start_date = prices.index[-1] + pd.Timedelta(days=1)
    sim_end_date = prices.index[-1] + pd.Timedelta(days=sim_days)

    actual_data = robust_download(ticker,
                                  sim_start_date.strftime("%Y-%m-%d"),
                                  sim_end_date.strftime("%Y-%m-%d"))
    if isinstance(actual_data.columns, pd.MultiIndex):
        actual_data.columns = actual_data.columns.get_level_values(0)

    actual_price = float("nan")
    if not actual_data.empty and "Close" in actual_data.columns:
        ap = actual_data["Close"].dropna()
        if not ap.empty:
            actual_price = float(ap.iloc[-1])

    runtime = time.time() - t0
    logging.info(f"Finished simulation for {ticker} in {runtime:.2f} seconds.")
    return pa.Table.from_pydict({
        "ticker": [ticker],
        "partial_mean": [part_mean],
        "num_paths": [num_paths],
        "actual_price": [actual_price],
        "runtime": [runtime]
    })
