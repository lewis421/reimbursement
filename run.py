import sys
import joblib
import pandas as pd
import numpy as np

# Load pre-trained stacking model
model_data = joblib.load('reimbursement_model_stacking_final.joblib')
rf_model = model_data['rf']
xgb_model = model_data['xgb']
nn_model = model_data['nn']
scaler = model_data['scaler']
meta_learner = model_data['meta_learner']

def engineer_features(trip_duration_days, miles_traveled, total_receipts_amount):
    data = {
        'trip_duration_days': trip_duration_days,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': total_receipts_amount
    }
    df = pd.DataFrame([data], index=[0])
    
    df['base_per_diem'] = df['trip_duration_days'] * 100
    df['five_day_bonus'] = (df['trip_duration_days'] == 5).astype(int)
    df['short_trip'] = (df['trip_duration_days'] <= 3).astype(int)
    df['medium_trip'] = ((df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 6)).astype(int)
    df['long_trip_8plus'] = (df['trip_duration_days'] >= 8).astype(int)
    df['long_trip_10plus'] = (df['trip_duration_days'] >= 10).astype(int)
    df['miles_first_100'] = np.minimum(df['miles_traveled'], 100)
    df['miles_over_100'] = np.maximum(0, df['miles_traveled'] - 100)
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['efficiency_low'] = (df['miles_per_day'] < 180).astype(int)
    df['efficiency_sweet'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
    df['efficiency_high'] = (df['miles_per_day'] > 220).astype(int)
    df['log_miles'] = np.log1p(df['miles_traveled'])
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
    df['quarter_end'] = 0
    df['q4_high_receipt'] = 0
    df['miles_times_days'] = df['miles_traveled'] * df['trip_duration_days']
    df['receipts_times_miles'] = df['total_receipts_amount'] * df['miles_traveled']
    df['receipts_per_day_times_miles_per_day'] = df['receipts_per_day'] * df['miles_per_day']
    df['receipts_per_mile'] = df['total_receipts_amount'] / np.maximum(df['miles_traveled'], 1)
    df['receipts_per_day_per_miles_per_day'] = df['receipts_per_day'] / np.maximum(df['miles_per_day'], 1)
    df['miles_per_day_times_days'] = df['miles_per_day'] * df['trip_duration_days']
    
    return df

def post_process(prediction, total_receipts_amount, trip_duration_days):
    # Rounding adjustment
    common_endings = [0.00, 0.10, 0.50]
    for ending in common_endings:
        if abs(prediction - (int(prediction) + ending)) <= 0.03:
            prediction = round(int(prediction) + ending, 2)
            break
    # Low receipt boost
    if total_receipts_amount < 50:
        prediction *= 1.05
    # High receipt penalty (tighter for long trips)
    if total_receipts_amount > 1500 and trip_duration_days >= 8:
        prediction *= 0.95
    elif total_receipts_amount > 1500:
        prediction *= 0.97
    return prediction

def main():
    if len(sys.argv) != 4:
        print("Usage: run.py trip_duration_days miles_traveled total_receipts_amount")
        sys.exit(1)
    
    try:
        trip_duration_days = int(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
    except ValueError:
        print("Inputs must be numbers")
        sys.exit(1)
    
    features = engineer_features(trip_duration_days, miles_traveled, total_receipts_amount)
    features_scaled = scaler.transform(features)
    
    rf_pred = rf_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]
    nn_pred = nn_model.predict(features_scaled)[0]
    
    meta_input = np.array([[rf_pred, xgb_pred, nn_pred]])
    prediction = meta_learner.predict(meta_input)[0]
    
    prediction = post_process(prediction, total_receipts_amount, trip_duration_days)
    print(f"{prediction:.2f}")

if __name__ == "__main__":
    main()