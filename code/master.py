#!/usr/bin/env python3
import os
import threading

import pyarrow as pa
import pyarrow.flight as flight


class MasterFlightServer(flight.FlightServerBase):
    def __init__(self, location, storage_dir="results", **kwargs):
        super().__init__(location, **kwargs)
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.counter = 0

    def do_put(self, context, descriptor, reader, writer):
        table = reader.read_all()
        with self.lock:
            self.counter += 1
            filename = os.path.join(self.storage_dir, f"result_{self.counter}.arrow")
            with pa.OSFile(filename, "wb") as sink:
                w = pa.RecordBatchFileWriter(sink, table.schema)
                w.write_table(table)
                w.close()
            print(f"Stored result {self.counter} in {filename}")
        return flight.PutResult()


def main():
    location = "grpc://0.0.0.0:8815"
    print(f"Master server at {location}")
    server = MasterFlightServer(location)
    server.serve()


if __name__ == "__main__":
    main()
