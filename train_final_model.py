# train_final_model.py
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import pickle

class FinalModelGenerator:
    def __init__(self, data_filepath='public_cases.json'):
        self.df = self._load_and_prepare_data(data_filepath)
        self._engineer_features()

    def _load_and_prepare_data(self, data_filepath):
        # (Loading and cleaning logic as before)
        try:
            with open(data_filepath, 'r') as f: data = json.load(f)
        except FileNotFoundError: return pd.DataFrame()
        valid_records, valid_outputs = [], []
        for item in data:
            try:
                record = {"trip_duration_days": int(item['input']['trip_duration_days']), "miles_traveled": int(float(item['input']['miles_traveled'])), "total_receipts_amount": float(item['input']['total_receipts_amount'])}
                valid_records.append(record); valid_outputs.append(item['expected_output'])
            except (ValueError, KeyError, TypeError): continue
        df = pd.DataFrame(valid_records)
        df['expected_output'] = valid_outputs
        return df

    def _engineer_features(self):
        # (Full feature engineering logic as before)
        if self.df.empty: return
        self.df['miles_per_day'] = (self.df['miles_traveled'] / self.df['trip_duration_days']).replace([np.inf, -np.inf], 0).fillna(0)
        self.df['spending_per_day'] = (self.df['total_receipts_amount'] / self.df['trip_duration_days']).replace([np.inf, -np.inf], 0).fillna(0)
        self.df['has_rounding_quirk'] = self.df['total_receipts_amount'].apply(lambda x: round(x * 100) % 100 in [49, 99]).astype(int)
        high_spending = self.df['spending_per_day'].quantile(0.75)
        low_spending = self.df['spending_per_day'].quantile(0.25)
        high_mileage = self.df['miles_per_day'].quantile(0.75)
        low_mileage = self.df['miles_per_day'].quantile(0.25)
        self.df['sweet_spot_combo'] = ((self.df['trip_duration_days'] == 5) & (self.df['miles_per_day'] >= 180) & (self.df['spending_per_day'] < 100)).astype(int)
        self.df['vacation_penalty'] = ((self.df['trip_duration_days'] >= 8) & (self.df['spending_per_day'] > high_spending)).astype(int)

    def _calculate_rule_based_estimate(self, df, params):
        # (Rule calculation logic as before)
        estimate = df['trip_duration_days'] * params['per_diem']
        miles = df['miles_traveled']
        mileage_reimbursement = np.where(miles <= 100, miles * params['mileage_rate_short'], (100 * params['mileage_rate_short']) + ((miles - 100) * params['mileage_rate_long']))
        estimate += mileage_reimbursement
        estimate += np.minimum(df['total_receipts_amount'] * params['receipt_percentage'], 1000)
        return estimate

    def run(self):
        if self.df.empty: return

        # 1. Find Optimal Rules on the full dataset
        print("Finding optimal rule parameters on full dataset...")
        best_params = {}
        lowest_mae = float('inf')
        for short_rate in [0.6, 0.7, 0.75, 0.8]:
            for long_rate in [0.4, 0.45, 0.5]:
                params = {'per_diem': 100, 'mileage_rate_short': short_rate, 'mileage_rate_long': long_rate, 'receipt_percentage': 0.8}
                estimates = self._calculate_rule_based_estimate(self.df, params)
                mae = mean_absolute_error(self.df['expected_output'], estimates)
                if mae < lowest_mae:
                    lowest_mae = mae
                    best_params = params
        with open('optimal_rules.json', 'w') as f: json.dump(best_params, f)
        print(f"Saved optimal rules: {best_params}")

        # 2. Train Error Model
        self.df['rule_based_estimate'] = self._calculate_rule_based_estimate(self.df, best_params)
        self.df['unexplained_error'] = self.df['expected_output'] - self.df['rule_based_estimate']
        
        features = [col for col in self.df.columns if col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'spending_per_day', 'has_rounding_quirk', 'sweet_spot_combo', 'vacation_penalty']]
        X = self.df[features]
        y = self.df['unexplained_error']
        
        print("Training final error model...")
        error_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        error_model.fit(X, y)
        with open('final_model.pkl', 'wb') as f: pickle.dump(error_model, f)
        print("Saved final model.")

        # 3. Find Final Corrections
        self.df['predicted_error'] = error_model.predict(X)
        self.df['final_model_error'] = self.df['expected_output'] - (self.df['rule_based_estimate'] + self.df['predicted_error'])
        
        corrections = {}
        corrections['rounding_quirk'] = self.df[self.df['has_rounding_quirk'] == 1]['final_model_error'].mean()
        corrections['sweet_spot_combo'] = self.df[self.df['sweet_spot_combo'] == 1]['final_model_error'].mean()
        corrections['vacation_penalty'] = self.df[self.df['vacation_penalty'] == 1]['final_model_error'].mean()
        with open('final_corrections.json', 'w') as f: json.dump(corrections, f)
        print(f"Saved final correction values: {corrections}")

if __name__ == '__main__':
    generator = FinalModelGenerator()
    generator.run()