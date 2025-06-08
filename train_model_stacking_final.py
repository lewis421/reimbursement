import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Create DataFrame
records = []
for case in data:
    record = {
        'trip_duration_days': case['input']['trip_duration_days'],
        'miles_traveled': case['input']['miles_traveled'],
        'total_receipts_amount': case['input']['total_receipts_amount'],
        'reimbursement': case['expected_output']
    }
    records.append(record)
df = pd.DataFrame(records)

# Feature engineering
def engineer_features(df):
    df = df.copy()
    # Per diem features
    df['base_per_diem'] = df['trip_duration_days'] * 100
    df['five_day_bonus'] = (df['trip_duration_days'] == 5).astype(int)
    df['short_trip'] = (df['trip_duration_days'] <= 3).astype(int)
    df['medium_trip'] = ((df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 6)).astype(int)
    df['long_trip_8plus'] = (df['trip_duration_days'] >= 8).astype(int)
    df['long_trip_10plus'] = (df['trip_duration_days'] >= 10).astype(int)
    
    # Mileage features
    df['miles_first_100'] = np.minimum(df['miles_traveled'], 100)
    df['miles_over_100'] = np.maximum(0, df['miles_traveled'] - 100)
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['efficiency_low'] = (df['miles_per_day'] < 180).astype(int)
    df['efficiency_sweet'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
    df['efficiency_high'] = (df['miles_per_day'] > 220).astype(int)
    df['log_miles'] = np.log1p(df['miles_traveled'])
    
    # Receipt features
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['low_receipt_penalty'] = (df['receipts_per_day'] < 30).astype(int)
    df['high_receipt_penalty_200'] = (df['receipts_per_day'] > 200).astype(int)
    df['high_receipt_penalty_300'] = (df['receipts_per_day'] > 300).astype(int)
    df['receipt_bin_0_50'] = (df['total_receipts_amount'] <= 50).astype(int)
    df['receipt_bin_50_200'] = ((df['total_receipts_amount'] > 50) & (df['total_receipts_amount'] <= 200)).astype(int)
    df['receipt_bin_200_600'] = ((df['total_receipts_amount'] > 200) & (df['total_receipts_amount'] <= 600)).astype(int)
    df['receipt_bin_600_800'] = ((df['total_receipts_amount'] > 600) & (df['total_receipts_amount'] <= 800)).astype(int)
    df['receipt_bin_800_plus'] = (df['total_receipts_amount'] > 800).astype(int)
    df['log_receipts'] = np.log1p(df['total_receipts_amount'])
    df['receipt_rounding'] = df['total_receipts_amount'].apply(lambda x: 1 if str(x).split('.')[-1][-2:] in ['49', '99'] else 0)
    df['vacation_penalty'] = ((df['trip_duration_days'] > 7) & (df['receipts_per_day'] > 90)).astype(int)
    df['receipt_cap'] = df['trip_duration_days'] * np.where(df['medium_trip'], 120, 90) * (1 + df['miles_per_day'] / 1000)
    
    # Temporal proxy
    df['quarter_end'] = (df.index % 4 == 0).astype(int)
    df['q4_high_receipt'] = ((df.index % 4 == 0) & (df['total_receipts_amount'] > 1000)).astype(int)
    
    # Interaction terms
    df['miles_times_days'] = df['miles_traveled'] * df['trip_duration_days']
    df['receipts_times_miles'] = df['total_receipts_amount'] * df['miles_traveled']
    df['receipts_per_day_times_miles_per_day'] = df['receipts_per_day'] * df['miles_per_day']
    df['receipts_per_mile'] = df['total_receipts_amount'] / np.maximum(df['miles_traveled'], 1)
    df['receipts_per_day_per_miles_per_day'] = df['receipts_per_day'] / np.maximum(df['miles_per_day'], 1)
    df['miles_per_day_times_days'] = df['miles_per_day'] * df['trip_duration_days']
    
    return df

# Prepare features and target
df_features = engineer_features(df)
X = df_features.drop('reimbursement', axis=1)
y = df_features['reimbursement']

# Train base models on full data
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
nn = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf.fit(X, y)
xgb.fit(X, y)
nn.fit(X_scaled, y)

# Generate base model predictions for meta-learner
full_preds = np.column_stack([
    rf.predict(X),
    xgb.predict(X),
    nn.predict(X_scaled)
])

# Train meta-learner (small Neural Network)
meta_learner = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=500, random_state=42)
meta_learner.fit(full_preds, y)

# Save models, scaler, and meta-learner
joblib.dump({'rf': rf, 'xgb': xgb, 'nn': nn, 'scaler': scaler, 'meta_learner': meta_learner}, 'reimbursement_model_stacking_final.joblib')

# Evaluate on full data
ensemble_pred = meta_learner.predict(full_preds)
mae = np.mean(np.abs(ensemble_pred - y))
exact_matches = np.sum(np.abs(ensemble_pred - y) <= 0.01)
close_matches = np.sum(np.abs(ensemble_pred - y) <= 1.00)
print(f"Stacking Ensemble (Final) - MAE: {mae:.2f}")
print(f"Exact matches: {exact_matches}/{len(y)} ({exact_matches/len(y)*100:.1f}%)")
print(f"Close matches: {close_matches}/{len(y)} ({close_matches/len(y)*100:.1f}%)")