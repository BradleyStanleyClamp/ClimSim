#!/bin/bash

BASE_DIR="/home/users/bradlesc/projects/ClimSim/logs/p2.1.1/9/testing/functional_test"

# Find the latest subdirectory by modification time
LATEST_DIR=$(ls -1dt "$BASE_DIR"/*/ | head -n 1)

# Define the log file path (one directory deeper)
LOGFILE=$(ls -1 "$LATEST_DIR"*/train_general.log 2>/dev/null | head -n 1)

if [[ -z "$LOGFILE" ]]; then
    echo "No train_general.log found under $LATEST_DIR"
    exit 1
fi

echo "Tailing: $LOGFILE"
tail -f "$LOGFILE"