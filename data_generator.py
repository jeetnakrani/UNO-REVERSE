import pandas as pd
import random
import time
from datetime import datetime

# Data pools
task_types = ['Cleaning', 'Repair', 'Inspection', 'Refill']
teams = ['Team Alpha', 'Team Bravo', 'Team Delta']
staff_names = ['Amit', 'Sara', 'John', 'Ravi', 'Priya']
statuses = ['Done', 'Missed']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# CSV structure
columns = [
    'task_id', 'task_type', 'team_name', 'date', 'start_time', 'duration (mins)',
    'assigned_staff', 'status', 'workload', 'resource_available', 'day_of_week'
]

# Create CSV if not present
try:
    df = pd.read_csv('task_data.csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)
    df.to_csv('task_data.csv', index=False)

task_id = len(df) + 1

while True:
    now = datetime.now()
    new_row = {
        'task_id': task_id,
        'task_type': random.choice(task_types),
        'team_name': random.choice(teams),
        'date': now.strftime('%Y-%m-%d'),
        'start_time': now.strftime('%H:%M:%S'),
        'duration (mins)': random.randint(10, 60),
        'assigned_staff': random.choice(staff_names),
        'status': random.choices(statuses, weights=[0.85, 0.15])[0],  # mostly Done
        'workload': random.randint(1, 10),
        'resource_available': round(random.uniform(0.3, 1.0), 2),  # Between 30% to 100%
        'day_of_week': weekdays[now.weekday()]
    }

    df = pd.read_csv('task_data.csv')
    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv('task_data.csv', index=False)

    print(f"✅ Added Task_ID: {task_id}")
    task_id += 1
    time.sleep(5)
