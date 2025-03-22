import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('facility_tasks.csv', parse_dates=['scheduled_time', 'start_time', 'completion_time'])

# 1. Standardize task status
df['status'] = df['status'].str.strip().str.title()

# 2. Handle missing values (use assignment instead of inplace)
df['start_time'] = df['start_time'].fillna(pd.NaT)
df['completion_time'] = df['completion_time'].fillna(pd.NaT)
df['feedback_score'] = df['feedback_score'].fillna(-1)  # Use -1 as placeholder for missing feedback
df['actual_duration'] = df['actual_duration'].fillna(-1)

# 3. Feature Engineering
df['hour_of_day'] = df['scheduled_time'].dt.hour
df['month'] = df['scheduled_time'].dt.month
df['time_to_start'] = (df['start_time'] - df['scheduled_time']).dt.total_seconds() / 60  # in minutes
df['time_to_start'] = df['time_to_start'].fillna(0)

# Task frequency features
df['day'] = df['scheduled_time'].dt.date
df['tasks_per_technician_per_day'] = df.groupby(['assigned_to', 'day'])['task_id'].transform('count')
df['avg_duration_per_task_type'] = df.groupby('task_type')['expected_duration'].transform('mean')
df['miss_rate_per_location'] = df.groupby('location')['was_missed'].transform('mean')

# Encode categorical variables
le_task = LabelEncoder()
le_asset = LabelEncoder()
le_location = LabelEncoder()
le_person = LabelEncoder()

df['task_type_enc'] = le_task.fit_transform(df['task_type'])
df['asset_type_enc'] = le_asset.fit_transform(df['asset_type'])
df['location_enc'] = le_location.fit_transform(df['location'])
df['assigned_to_enc'] = le_person.fit_transform(df['assigned_to'])

# Encode priority
df['priority_enc'] = df['priority'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Final list of features to be used in ML
processed_features = [
    'expected_duration', 'actual_duration', 'technician_experience',
    'asset_age', 'hour_of_day', 'month', 'time_to_start',
    'tasks_per_technician_per_day', 'avg_duration_per_task_type',
    'miss_rate_per_location', 'task_type_enc', 'asset_type_enc',
    'location_enc', 'assigned_to_enc', 'priority_enc'
]

# Export clean, preprocessed dataset (before SMOTE)
final_df = df[['was_missed'] + processed_features]
final_df.to_csv("processed_facility_data.csv", index=False)
print("✅ Preprocessed data saved to 'processed_facility_data.csv'")
