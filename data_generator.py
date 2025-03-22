import pandas as pd
import random
import time
from datetime import datetime, timedelta
import os

# --- Data Pools ---
task_types = ['Cleaning', 'Repair', 'Inspection', 'Refill']
teams = ['Team Alpha', 'Team Bravo', 'Team Delta']
staff_names = ['Amit', 'Sara', 'John', 'Ravi', 'Priya']
statuses = ['Done', 'Missed']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# --- CSV Columns ---
columns = [
    'task_id', 'task_type', 'team_name', 'date', 'start_time', 'duration (mins)',
    'assigned_staff', 'status', 'workload', 'resource_available', 'day_of_week'
]

# --- Step 1: Create file and prefill with 2000 rows (if not exists) ---
csv_file = "task_data.csv"

if not os.path.exists(csv_file):
    print("🔄 Generating initial dataset...")

    data = []
    for i in range(1, 2001):
        random_days_ago = random.randint(0, 59)
        task_date = datetime.now() - timedelta(days=random_days_ago)
        day_of_week = weekdays[task_date.weekday()]
        new_row = {
            'task_id': i,
            'task_type': random.choice(task_types),
            'team_name': random.choice(teams),
            'date': task_date.strftime('%Y-%m-%d'),
            'start_time': task_date.strftime('%H:%M:%S'),
            'duration (mins)': random.randint(10, 60),
            'assigned_staff': random.choice(staff_names),
            'status': random.choices(statuses, weights=[0.6, 0.4])[0],
            'workload': random.randint(1, 10),
            'resource_available': round(random.uniform(0.3, 1.0), 2),
            'day_of_week': day_of_week
        }
        data.append(new_row)

    df_init = pd.DataFrame(data, columns=columns)
    df_init.to_csv(csv_file, index=False)
    print("✅ Initial dataset created with 2000 entries.")

# --- Step 2: Start appending new tasks every 60 seconds ---
print("🚀 Starting real-time task generation every 60 seconds...")

task_id = pd.read_csv(csv_file).shape[0] + 1

while True:
    now = datetime.now()
    day_of_week = weekdays[now.weekday()]
    new_row = {
        'task_id': task_id,
        'task_type': random.choice(task_types),
        'team_name': random.choice(teams),
        'date': now.strftime('%Y-%m-%d'),
        'start_time': now.strftime('%H:%M:%S'),
        'duration (mins)': random.randint(10, 60),
        'assigned_staff': random.choice(staff_names),
        'status': random.choices(statuses, weights=[0.85, 0.15])[0],
        'workload': random.randint(1, 10),
        'resource_available': round(random.uniform(0.3, 1.0), 2),
        'day_of_week': day_of_week
    }

    df = pd.read_csv(csv_file)
    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(csv_file, index=False)

    print(f"🆕 Added Task_ID: {task_id} at {now.strftime('%H:%M:%S')}")
    task_id += 1
    time.sleep(60)  # 60 seconds pause
