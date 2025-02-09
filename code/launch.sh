#!/bin/bash

TOTAL_PATHS=1000000
TICKERS="AAPL,AMZN,NFLX,GOOGL,META"

NUM_WORKERS=$(($SLURM_NTASKS - 1))
PATHS_PER_WORKER=$(($TOTAL_PATHS / $NUM_WORKERS))


echo "Total simulation paths: $TOTAL_PATHS" 
echo "Number of workers: $NUM_WORKERS" 
echo "Paths per worker: $PATHS_PER_WORKER" 

if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Master rank 0 starting on host $(hostname)."
    python master.py --tickers "$TICKERS" --num-workers "$NUM_WORKERS" --verbose
else
    MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    flight_endpoint="grpc://$MASTER_NODE:8815"
    echo "Worker rank=$SLURM_PROCID on host $(hostname). Connecting to $flight_endpoint."
    python worker.py "$flight_endpoint" --tickers "$TICKERS" --paths "$PATHS_PER_WORKER" --verbose
fi


