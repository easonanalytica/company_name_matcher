# test_comparison.py

import time
from company_name_matcher import CompanyNameMatcher
from rapidfuzz import fuzz
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the test data
df = pd.read_csv('test_data.csv')

# Initialize our CompanyNameMatcher
matcher = CompanyNameMatcher("models/fine_tuned_model")

def test_company_name_matcher():
    start_time = time.time()
    predictions = []

    for _, row in df.iterrows():
        similarity = matcher.compare_companies(row['Name_x'], row['Name_y'])
        predictions.append(1 if similarity >= 0.9 else 0)  # You can adjust this threshold

    end_time = time.time()

    return predictions, end_time - start_time

def test_rapidfuzz():
    start_time = time.time()
    predictions = []

    for _, row in df.iterrows():
        similarity = fuzz.ratio(row['Name_x'], row['Name_y']) / 100
        predictions.append(1 if similarity >= 0.9 else 0)  # You can adjust this threshold

    end_time = time.time()

    return predictions, end_time - start_time

# Run tests
cnm_predictions, cnm_time = test_company_name_matcher()
rf_predictions, rf_time = test_rapidfuzz()

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

cnm_metrics = calculate_metrics(df['Targets'], cnm_predictions)
rf_metrics = calculate_metrics(df['Targets'], rf_predictions)

# Print results
print("Company Name Matcher Results:")
print(f"Time taken: {cnm_time:.4f} seconds")
print(f"Metrics: {cnm_metrics}")
print("\nRapidFuzz Results:")
print(f"Time taken: {rf_time:.4f} seconds")
print(f"Metrics: {rf_metrics}")
