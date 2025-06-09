import json
import hashlib
import numpy as np
import pandas as pd

# Load pc.txt
with open('public_cases.json', 'r') as f:
    data = json.load(f)

#Create list of 1d dicts from list of 2d dicts
ddata = [{'trip_duration_days': case['input']['trip_duration_days'], 'miles_traveled': case['input']['miles_traveled'], 'total_receipts_amount': case['input']['total_receipts_amount'],'expected_output': case['expected_output']} for case in data]
print(ddata[:5])  # Print first five cases for verification

df = pd.DataFrame(ddata)
print(df.head(),df.tail())


# Statistics for all columns
print(df.describe(include='all'))

# Verify number of entries
print(f"Number of entries: {len(data)}")

# Print first five cases
print("\nFirst five cases:")
for i in range(5):
    case = data[i]
    print(f"Case {i+1}:")
    print(f"  Inputs: {case['input']}")
    print(f"  Expected Output: {case['expected_output']}")

# Print last five cases
print("\nLast five cases:")
for i in range(-5, 0):
    case = data[i]
    print(f"Case {len(data)+i+1}:")
    print(f"  Inputs: {case['input']}")
    print(f"  Expected Output: {case['expected_output']}")

# Compute SHA256 checksum
with open('public_cases.json', 'rb') as f:
    checksum = hashlib.sha256(f.read()).hexdigest()
print(f"\nSHA256 Checksum: {checksum}")

# Basic statistics
reimbursements = [case['expected_output'] for case in data]
days = [case['input']['trip_duration_days'] for case in data]
miles = [case['input']['miles_traveled'] for case in data]
receipts = [case['input']['total_receipts_amount'] for case in data]

print(f"\nDataset Statistics:")
print(f"  Reimbursement - Min: ${min(reimbursements):.2f}, Max: ${max(reimbursements):.2f}, Mean: ${np.mean(reimbursements):.2f}")
print(f"  Days - Min: {min(days)}, Max: {max(days)}, Mean: {np.mean(days):.2f}")
print(f"  Miles - Min: {min(miles):.2f}, Max: {max(miles):.2f}, Mean: {np.mean(miles):.2f}")
print(f"  Receipts - Min: ${min(receipts):.2f}, Max: ${max(receipts):.2f}, Mean: ${np.mean(receipts):.2f}")