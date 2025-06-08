#!/bin/bash

# Black Box Challenge - Implementation Runner
# Executes reimbursement_engine.py with input parameters as floats

if [ $# -ne 3 ]; then
    echo "Error: Exactly 3 arguments required: trip_duration_days miles_traveled total_receipts_amount" >&2
    exit 1
fi

for arg in "$@"; do
    if ! [[ $arg =~ ^[0-9]+(\.[0-9]+)?$|^[0-9]+$ ]]; then
        echo "Error: All arguments must be numbers" >&2
        exit 1
    fi
done

# Convert arguments to floats using printf
days=$(printf "%.1f" "$1")
miles=$(printf "%.1f" "$2")
receipts=$(printf "%.2f" "$3")
# printf "Running reimbursement_engine.py with:\n  Days: %s\n  Miles: %s\n  Receipts: %s\n" "$days" "$miles" "$receipts"
python3 run.py "$days" "$miles" "$receipts"