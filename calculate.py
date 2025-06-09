# calculate.py
import sys
import pandas as pd
import numpy as np
import pickle
import json

# --- Load All Final Artifacts ---
try:
    with open('optimal_rules.json', 'r') as f: rules = json.load(f)
    with open('final_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('final_corrections.json', 'r') as f: corrections = json.load(f)
except FileNotFoundError as e:
    print(f"Error: A required file was not found ({e}). Please run train_final_model.py first.", file=sys.stderr)
    sys.exit(1)

# --- Get Inputs ---
try:
    days = int(sys.argv[1])
    miles = int(float(sys.argv[2]))
    receipts = float(sys.argv[3])
except (ValueError, IndexError):
    sys.exit(1)

# --- Feature Engineering ---
df = pd.DataFrame([{'trip_duration_days': days, 'miles_traveled': miles, 'total_receipts_amount': receipts}])
df['miles_per_day'] = (df['miles_traveled'] / df['trip_duration_days']).replace([np.inf, -np.inf], 0).fillna(0)
df['spending_per_day'] = (df['total_receipts_amount'] / df['trip_duration_days']).replace([np.inf, -np.inf], 0).fillna(0)
df['has_rounding_quirk'] = df['total_receipts_amount'].apply(lambda x: round(x * 100) % 100 in [49, 99]).astype(int)
# In a real scenario, these quantiles would be saved from the training set. We use approximations here.
high_spending = df['spending_per_day'].quantile(0.75) if 'spending_per_day' in df else 150 
df['sweet_spot_combo'] = ((df['trip_duration_days'] == 5) & (df['miles_per_day'] >= 180) & (df['spending_per_day'] < 100)).astype(int)
df['vacation_penalty'] = ((df['trip_duration_days'] >= 8) & (df['spending_per_day'] > high_spending)).astype(int)

# --- Final Calculation ---
# 1. Start with the optimized rule-based estimate
estimate = days * rules.get('per_diem', 100)
estimate += np.where(miles <= 100, miles * rules.get('mileage_rate_short', 0.75), (100 * rules.get('mileage_rate_short', 0.75)) + ((miles - 100) * rules.get('mileage_rate_long', 0.5)))
estimate += np.minimum(receipts * rules.get('receipt_percentage', 0.8), 1000)

# 2. Add the ML model's prediction for the unexplained error
features = [col for col in df.columns if col in model.feature_names_in_]
predicted_error = model.predict(df[features])[0]
estimate += predicted_error

# 3. Add the final hard-coded corrections
if df['sweet_spot_combo'].iloc[0]:
    estimate += corrections.get('sweet_spot_combo', 0)
elif df['vacation_penalty'].iloc[0]:
    estimate += corrections.get('vacation_penalty', 0)
if df['has_rounding_quirk'].iloc[0]:
    estimate += corrections.get('rounding_quirk', 0)

# --- Output Final Result ---
print(f"{estimate:.2f}")