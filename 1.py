import streamlit as st
import pandas as pd
from datetime import timedelta
from ml_model import train_predictive_model
from forecasting import forecast_missed_tasks
from anomaly_detection import detect_anomalies

st.set_page_config(page_title="Facility Tasks Dashboard", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("facility_tasks.csv", parse_dates=['scheduled_time', 'start_time', 'completion_time'])
    df['DATE'] = df['scheduled_time'].dt.date
    df['was_missed'] = df['was_missed'].astype(int)
    df['actual_duration'] = df['actual_duration'].fillna(0)
    df['feedback_score'] = df['feedback_score'].fillna(df['feedback_score'].mean())
    return df

df = load_data()

# Sidebar filters
with st.sidebar:
    st.header("⚙️ Dashboard Filters")
    start_date = st.date_input("Start Date", df['DATE'].min())
    end_date = st.date_input("End Date", df['DATE'].max())
    teams = st.multiselect("Assigned Team", df['assigned_to'].unique(), default=df['assigned_to'].unique())
    task_types = st.multiselect("Task Type", df['task_type'].unique(), default=df['task_type'].unique())

# Filter data based on sidebar selections
filtered_df = df[
    (df['DATE'] >= start_date) &
    (df['DATE'] <= end_date) &
    (df['assigned_to'].isin(teams)) &
    (df['task_type'].isin(task_types))
]

# Aggregate metrics
total_tasks = len(filtered_df)
missed_tasks = filtered_df['was_missed'].sum()
total_actual_hours = filtered_df['actual_duration'].sum() / 60  # in hours
avg_feedback_score = filtered_df['feedback_score'].mean()

# Dashboard Title and Key Metrics
st.title("🚀 Smart Facility Task Dashboard")

cols = st.columns(4)
cols[0].metric("🗂️ Total Tasks", total_tasks)
cols[1].metric("❌ Missed Tasks", missed_tasks)
cols[2].metric("⏳ Actual Work Hours", f"{total_actual_hours:.2f} hrs")
cols[3].metric("⭐ Avg. Feedback", f"{avg_feedback_score:.2f}/5")

# Missed Tasks Over Time Visualization
st.subheader("📉 Missed Tasks Over Time")
missed_over_time = filtered_df.groupby('DATE')['was_missed'].sum().reset_index()
st.bar_chart(missed_over_time.set_index('DATE'))

# Feedback Score Trend Visualization
st.subheader("🌟 Feedback Score Trend")
feedback_trend = filtered_df.groupby('DATE')['feedback_score'].mean().reset_index()
st.line_chart(feedback_trend.set_index('DATE'))

# Technician Performance Table
st.subheader("👨‍🔧 Technician Performance")
tech_perf = filtered_df.groupby('assigned_to').agg({
    'was_missed': 'sum',
    'feedback_score': 'mean',
    'task_id': 'count'
}).rename(columns={'was_missed':'Missed Tasks', 'feedback_score':'Avg Feedback', 'task_id':'Total Tasks'})
st.dataframe(tech_perf.sort_values(by='Missed Tasks', ascending=False))

# Location Analysis
st.subheader("📍 Location-wise Missed Tasks")
loc_analysis = filtered_df.groupby('location')['was_missed'].sum().sort_values(ascending=False)
st.bar_chart(loc_analysis)

# Asset Type Analysis
st.subheader("🛠️ Asset Type Missed Tasks")
asset_perf = filtered_df.groupby('asset_type')['was_missed'].sum().sort_values(ascending=False)
st.bar_chart(asset_perf)

# Predictive Modeling Integration
st.subheader("🔮 Predictive Modeling - Task Miss Prediction")
with st.spinner('Training model...'):
    model, report = train_predictive_model(df)
st.success("Model trained successfully!")
st.write("**Classification Report:**")
st.dataframe(pd.DataFrame(report).transpose())

# Forecasting with Prophet
st.subheader("📅 Forecasting Missed Tasks (Next 30 Days)")
with st.spinner('Generating forecast...'):
    prophet_model, forecast = forecast_missed_tasks(df)
fig_forecast = prophet_model.plot(forecast)
st.pyplot(fig_forecast)

# Anomaly Detection
st.subheader("🚨 Anomaly Detection (Unusual Task Misses)")
anomalies = detect_anomalies(df)
if not anomalies.empty:
    st.warning("Anomalies Detected!")
    st.dataframe(anomalies[['DATE', 'was_missed']])
else:
    st.success("✅ No significant anomalies detected.")

# Detailed Task Data Table
st.subheader("🔍 Detailed Task Data")
st.dataframe(filtered_df[['task_id','task_type','assigned_to','location','priority','DATE','was_missed','actual_duration','feedback_score']])
