import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Plot aggregated results for all tickers")
    parser.add_argument("--tickers", type=str, required=True,
                        help="Space-separated list of tickers, e.g., 'AAPL AMZN NFLX GOOGL META'")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING,
                            format="%(asctime)s %(levelname)s: %(message)s")
    tickers = args.tickers.split()
    summaries = []
    for ticker in tickers:
        csv_file = f"aggregated_{ticker}.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            summaries.append(df)
        else:
            logging.error(f"CSV file for {ticker} not found: {csv_file}")
    if not summaries:
        logging.error("No aggregated CSV files found.")
        return
    all_summary = pd.concat(summaries, ignore_index=True)
    logging.info("Aggregated summary for all tickers:")
    logging.info(all_summary)
    fig, ax = plt.subplots()
    x = all_summary["ticker"]
    width = 0.35
    ax.bar(x - width/2, all_summary["aggregated_mean"], width, label="Predicted")
    ax.bar(x + width/2, all_summary["actual_price"], width, label="Actual")
    ax.set_ylabel("Price")
    ax.set_title("Aggregated Simulation Prediction vs Actual Price for FAANG")
    ax.legend()
    plt.savefig("aggregated_faang_results.png")
    plt.show()

if __name__ == "__main__":
    main()
