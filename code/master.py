import threading
import time
import logging
import argparse
import os

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc
import pandas as pd

class PersistentMasterServer(flight.FlightServerBase):
    def __init__(self, location, tickers, num_workers, **kwargs):
        super().__init__(location, **kwargs)
        self.lock = threading.Lock()
        self.tickers = tickers
        self.num_workers = num_workers
        self.partial_counts = {t: 0 for t in self.tickers}
        self.storage = {t: [] for t in self.tickers}  
        self.done_tickers = set()
        os.makedirs("results", exist_ok=True)

    def do_put(self, context, descriptor, reader, writer):
        if not descriptor.path:
            logging.error("No descriptor path found. Can't identify ticker.")
            return None
        ticker = descriptor.path[0].decode("utf-8")
        logging.info(f"Received partial do_put for ticker={ticker} from a worker.")
        table = reader.read_all()

        with self.lock:
            if ticker not in self.storage:
                self.storage[ticker] = []
                self.partial_counts[ticker] = 0
            self.storage[ticker].append(table)
            self.partial_counts[ticker] += 1
            count = self.partial_counts[ticker]
            logging.info(f"{ticker}: partial count is now {count} of {self.num_workers}.")
            if count == self.num_workers:
                self.aggregate_and_save(ticker)

    def aggregate_and_save(self, ticker):
        logging.info(f"Aggregating partials for {ticker}.")
        tables = self.storage[ticker]
        df_all = pd.concat([tbl.to_pandas() for tbl in tables], ignore_index=True)
        total_paths = df_all["num_paths"].sum()
        aggregated_mean = float("nan")
        if total_paths > 0:
            weighted_sum = (df_all["partial_mean"] * df_all["num_paths"]).sum()
            aggregated_mean = weighted_sum / total_paths
        if "actual_price" in df_all.columns and not df_all["actual_price"].empty:
            actual_price = df_all["actual_price"].iloc[0]
        else:
            actual_price = float("nan")
        error = aggregated_mean - actual_price
        logging.info(f"{ticker}: aggregated_mean={aggregated_mean:.3f}, actual={actual_price:.3f}, error={error:.3f}")
        out_path = f"results/aggregated_{ticker}_for_{self.num_workers}_workers.csv"
        pd.DataFrame([{
            "ticker": ticker,
            "aggregated_mean": aggregated_mean,
            "actual_price": actual_price,
            "error": error
        }]).to_csv(out_path, index=False)
        logging.info(f"Saved aggregated result for {ticker} to {out_path}.")
        self.done_tickers.add(ticker)

    def all_done(self):
        return set(self.tickers) == self.done_tickers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="AAPL,AMZN,NFLX,GOOGL,META",
                        help="Comma-separated ticker list.")
    parser.add_argument("--num-workers", type=int, default=3,
                        help="Number of workers expected.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    tickers = args.tickers.split(",")
    location = "grpc://0.0.0.0:8815"
    server = PersistentMasterServer(location, tickers, args.num_workers)
    logging.info(f"Starting persistent Master server on {location}. Tickers={tickers}. Expecting {args.num_workers} partial results per ticker.")

    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()

    logging.info("Master server running. Will exit once all tickers are done.")
    while True:
        time.sleep(3)
        with server.lock:
            if server.all_done():
                logging.info("All tickers are done. Stopping server.")
                break

    server.shutdown()
    logging.info("Server shutdown complete. Exiting.")
    return 0

if __name__ == "__main__":
    main()
