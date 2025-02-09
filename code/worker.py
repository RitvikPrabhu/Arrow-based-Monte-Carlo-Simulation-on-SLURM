import argparse
import logging
import sys
import time

import pyarrow as pa
import pyarrow.flight as flight

from simulate import simulate_ticker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("flight_endpoint", help="Flight server endpoint, e.g. grpc://hostname:8815")
    parser.add_argument("--tickers", default="AAPL,AMZN,NFLX,GOOGL,META",
                        help="Comma-separated tickers list.")
    parser.add_argument("--paths", type=int, default=10000,
                        help="Number of Monte Carlo paths for each ticker.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    tickers = args.tickers.split(",")
    logging.info(f"Worker connecting to {args.flight_endpoint}. Tickers={tickers}, paths={args.paths}.")
    try:
        client = flight.FlightClient(args.flight_endpoint)
        logging.info("Connected to the Flight server.")
    except Exception as e:
        logging.error(f"Failed to connect to Flight server {args.flight_endpoint}: {e}")
        sys.exit(1)

    for ticker in tickers:
        logging.info(f"Simulating ticker={ticker} with paths={args.paths}.")
        table = simulate_ticker(ticker, num_paths=args.paths)
        descriptor = flight.FlightDescriptor.for_path(ticker.encode("utf-8"))
        try:
            writer, _ = client.do_put(descriptor, table.schema)
            writer.write_table(table)
            writer.close()
            logging.info(f"Sent partial result for {ticker}.")
        except Exception as e:
            logging.error(f"Error sending partial for {ticker}: {e}")
            sys.exit(1)
        time.sleep(1)

    logging.info("Worker finished sending partials for all tickers.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
