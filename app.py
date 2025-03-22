import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os

# Page setup
st.set_page_config(page_title="Smart Task Dashboard", layout="wide")
st.title("📊 Real-Time Facility Task Dashboard")

# Try to load the dataset
file_path = 'task_data.csv'

if not os.path.exists(file_path):
    st.error("❌ task_data.csv not found. Please run the data generator.")
    st.stop()

try:
    df = pd.read_csv(file_path)
    if df.empty:
        st.warning("⚠️ The dataset is currently empty. Wait for data generation.")
        st.stop()
except Exception as e:
    st.error(f"🚨 Error loading data: {e}")
    st.stop()

# Auto-refresh every 5 seconds
time.sleep(5)
st.experimental_rerun()

# Last update time
last_row = df.tail(1).iloc[0]
st.caption(f"🕒 Last Updated: {last_row['date']} {last_row['start_time']}")

# Filters
with st.sidebar:
    st.header("🔍 Filters")
    task_type = st.selectbox("Task Type", ["All"] + sorted(df["task_type"].unique().tolist()))
    team_name = st.selectbox("Team", ["All"] + sorted(df["team_name"].unique().tolist()))
    staff = st.selectbox("Assigned Staff", ["All"] + sorted(df["assigned_staff"].unique().tolist()))
    status = st.selectbox("Status", ["All"] + sorted(df["status"].unique().tolist()))

# Apply filters
filtered_df = df.copy()

if task_type != "All":
    filtered_df = filtered_df[filtered_df["task_type"] == task_type]
if team_name != "All":
    filtered_df = filtered_df[filtered_df["team_name"] == team_name]
if staff != "All":
    filtered_df = filtered_df[filtered_df["assigned_staff"] == staff]
if status != "All":
    filtered_df = filtered_df[filtered_df["status"] == status]

# Show latest entries
st.subheader("📝 Recent Task Logs")
st.dataframe(filtered_df.tail(20), use_container_width=True)

# Visualizations
st.subheader("📈 Missed Tasks by Team")
missed = filtered_df[filtered_df["status"] == "Missed"]
if not missed.empty:
    missed_by_team = missed["team_name"].value_counts()
    st.bar_chart(missed_by_team)
else:
    st.info("✅ No missed tasks in current view.")

st.subheader("📊 Task Distribution by Type")
task_counts = filtered_df["task_type"].value_counts()
st.bar_chart(task_counts)

st.subheader("🔬 Workload vs Resource Availability")
st.scatter_chart(filtered_df.tail(50), x="workload", y="resource_available")
