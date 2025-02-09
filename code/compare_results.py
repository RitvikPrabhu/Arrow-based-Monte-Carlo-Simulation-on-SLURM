#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


def load_results(results_dir="results"):
    files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".arrow")
    ]
    dfs = []
    for f in files:
        with pa.memory_map(f, "r") as source:
            reader = ipc.RecordBatchFileReader(source)
            table = reader.read_all()
            dfs.append(table.to_pandas())
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    df = load_results()
    if df.empty:
        print("No results found.")
        return
    grouped = df.groupby("ticker").agg(
        {
            "predicted_mean": "mean",
            "actual_price": "mean",
            "error": "mean",
            "runtime": "mean",
        }
    )
    print(grouped)
    grouped["abs_error"] = grouped["error"].abs()
    ax = grouped["abs_error"].plot(
        kind="bar", title="Distributed: Average Absolute Error per Ticker"
    )
    ax.set_ylabel("Absolute Error")
    plt.savefig("distributed_errors.png")
    plt.show()


if __name__ == "__main__":
    main()
