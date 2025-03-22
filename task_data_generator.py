import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import time
import os

# -------------------------------
# Initial Setup
# -------------------------------

task_types = ['HVAC Repair', 'Cleaning', 'Inspection', 'Plumbing', 'Electrical']
asset_types = ['Generator', 'Fire Alarm', 'Light Fixture', 'AC Unit', 'Pump']
locations = ['Building A - Floor 1', 'Building A - Floor 2', 'Building B - Floor 1', 'Warehouse', 'Lobby']
technicians = ['Tech_01', 'Tech_02', 'Tech_03', 'Tech_04', 'Tech_05']
priority_levels = ['High', 'Medium', 'Low']
status_options = ['Scheduled', 'In Progress', 'Completed', 'Missed']
delay_reasons = ['Resource Unavailable', 'Weather Delay', 'Late Assignment', 'Access Denied', '']

columns = [
    'task_id', 'task_type', 'assigned_to', 'asset_type', 'location', 'priority',
    'scheduled_time', 'start_time', 'completion_time', 'status', 'delay_reason',
    'feedback_score', 'was_missed', 'expected_duration', 'actual_duration',
    'technician_experience', 'asset_age', 'day_of_week', 'is_weekend',
    'prev_misses_by_person', 'prev_misses_by_asset'
]

def generate_task(task_id, scheduled_time):
    task_type = random.choice(task_types)
    asset_type = random.choice(asset_types)
    location = random.choice(locations)
    assigned_to = random.choice(technicians)
    priority = random.choice(priority_levels)

    expected_duration = random.randint(30, 180)

    status = np.random.choice(status_options, p=[0.15, 0.10, 0.6, 0.15])
    delay_reason = random.choice(delay_reasons) if status == 'Missed' else ''

    start_time = scheduled_time + timedelta(minutes=random.randint(-15, 30)) if status != 'Scheduled' else pd.NaT
    actual_duration = expected_duration + random.randint(-10, 40) if status == 'Completed' else np.nan
    completion_time = start_time + timedelta(minutes=actual_duration) if pd.notnull(start_time) and status == 'Completed' else pd.NaT

    feedback_score = random.randint(3, 5) if status == 'Completed' else np.nan
    was_missed = status == 'Missed'

    technician_experience = random.randint(1, 10)
    asset_age = random.randint(1, 15)
    day_of_week = scheduled_time.strftime("%A")
    is_weekend = day_of_week in ['Saturday', 'Sunday']
    prev_misses_by_person = random.randint(0, 10)
    prev_misses_by_asset = random.randint(0, 15)

    return [
        task_id, task_type, assigned_to, asset_type, location, priority,
        scheduled_time, start_time, completion_time, status, delay_reason,
        feedback_score, was_missed, expected_duration, actual_duration,
        technician_experience, asset_age, day_of_week, is_weekend,
        prev_misses_by_person, prev_misses_by_asset
    ]

# -------------------------------
# Step 1: Create initial dataset
# -------------------------------

file_path = 'facility_tasks.csv'
if not os.path.exists(file_path):
    print("Creating initial dataset with 2500 tasks...")
    initial_data = []
    start_date = datetime.now() - timedelta(days=60)
    for i in range(1, 2501):
        random_day_offset = random.randint(0, 59)
        task_date = start_date + timedelta(days=random_day_offset)
        task = generate_task(f"TASK_{i:05d}", task_date)
        initial_data.append(task)
    df = pd.DataFrame(initial_data, columns=columns)
    df.to_csv(file_path, index=False)
    print("Initial dataset created!")
else:
    df = pd.read_csv(file_path)
    print("Loaded existing dataset!")

# -------------------------------
# Step 2: Real-time data generator
# -------------------------------

task_count = len(df)

while True:
    task_count += 1
    new_task = generate_task(f"TASK_{task_count:05d}", datetime.now())
    new_row = pd.DataFrame([new_task], columns=columns)
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_path, index=False)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Appended TASK_{task_count:05d}")
    time.sleep(60)  # wait for 60 seconds
