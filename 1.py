import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from ml_model import train_predictive_model
from forecasting import forecast_missed_tasks
from anomaly_detection import detect_anomalies
import shap
from sklearn.cluster import KMeans

st.set_page_config(page_title="Smart Facility Tasks Dashboard", layout="wide")

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

filtered_df = df[
    (df['DATE'] >= start_date) &
    (df['DATE'] <= end_date) &
    (df['assigned_to'].isin(teams)) &
    (df['task_type'].isin(task_types))
]

# Aggregate metrics
total_tasks = len(filtered_df)
missed_tasks = filtered_df['was_missed'].sum()
total_actual_hours = filtered_df['actual_duration'].sum() / 60
avg_feedback_score = filtered_df['feedback_score'].mean()

# Dashboard Title & Key Metrics
st.title("🚀 Smart Facility Task Dashboard")

cols = st.columns(4)
cols[0].metric("🗂️ Total Tasks", total_tasks)
cols[1].metric("❌ Missed Tasks", missed_tasks)
cols[2].metric("⏳ Actual Work Hours", f"{total_actual_hours:.2f} hrs")
cols[3].metric("⭐ Avg. Feedback", f"{avg_feedback_score:.2f}/5")

# Missed Tasks Visualization
st.subheader("📉 Missed Tasks Over Time")
missed_over_time = filtered_df.groupby('DATE')['was_missed'].sum().reset_index()
st.bar_chart(missed_over_time.set_index('DATE'))

# Feedback Score Trend
st.subheader("🌟 Feedback Score Trend")
feedback_trend = filtered_df.groupby('DATE')['feedback_score'].mean().reset_index()
st.line_chart(feedback_trend.set_index('DATE'))

# Technician Performance
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

# Predictive Modeling
st.subheader("🔮 Predictive Modeling - Task Miss Prediction")
with st.spinner('Training predictive model...'):
    model, report = train_predictive_model(df)
st.write("Classification Report:")
st.dataframe(pd.DataFrame(report).transpose())



# Forecasting with Prophet
st.subheader("📅 Forecasting Missed Tasks (Next 30 Days)")
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
    st.success("No significant anomalies detected.")

# Resource Optimization
st.subheader("🛠️ Resource Optimization Suggestions")
avg_daily_missed = filtered_df.groupby('DATE')['was_missed'].sum().mean()
recommended_staffing = int(np.ceil(avg_daily_missed / 5))
st.info(f"Optimal staffing: **{recommended_staffing} Technicians/day**")

# Adaptive Scheduling Alerts
st.subheader("🚦 Adaptive Scheduling Alerts")
today = filtered_df['DATE'].max()
today_missed = filtered_df[filtered_df['DATE'] == today]['was_missed'].sum()
if today_missed > avg_daily_missed:
    st.warning(f"⚠️ Alert: Today's missed tasks ({today_missed}) exceed average ({avg_daily_missed:.1f}). Consider adjusting schedules.")
else:
    st.success("✅ Today's performance is normal.")

# Auto-Clustering of Task Types
st.subheader("📊 Auto-Clustering of Task Types")
X_cluster = filtered_df[['expected_duration', 'technician_experience', 'asset_age']].fillna(0)
filtered_df['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_cluster)
cluster_summary = filtered_df.groupby('cluster')['was_missed'].mean()
st.bar_chart(cluster_summary)

# ===== Interactive Task Miss Prediction =====

st.sidebar.header("🔮 Interactive Task Prediction")

# User Inputs for Prediction
user_duration = st.sidebar.number_input("Expected Duration (minutes)", min_value=5, max_value=240, value=30)
user_experience = st.sidebar.number_input("Technician Experience (years)", min_value=0, max_value=30, value=3)
user_asset_age = st.sidebar.number_input("Asset Age (years)", min_value=0, max_value=50, value=5)
user_prev_misses_person = st.sidebar.number_input("Previous Misses by Technician", min_value=0, max_value=50, value=1)
user_prev_asset_misses = st.sidebar.number_input("Previous Misses by Asset", min_value=0, max_value=50, value=1)
user_priority = st.sidebar.selectbox("Priority", ['Low', 'Medium', 'High'])

# Create user input dataframe correctly
user_input_df = pd.DataFrame({
    'expected_duration': [user_duration],
    'technician_experience': [user_experience],
    'asset_age': [user_asset_age],
    'prev_misses_by_person': [user_prev_misses_person],
    'prev_misses_by_asset': [user_prev_asset_misses],
    'priority': [user_priority]
})

# Prepare input dataframe consistently with trained model
full_df = pd.get_dummies(df[['expected_duration', 'technician_experience', 'asset_age', 
                             'prev_misses_by_person', 'prev_misses_by_asset', 'priority']], drop_first=True)

user_input_processed = pd.get_dummies(user_input_df, drop_first=True)
user_input_processed = user_input_processed.reindex(columns=full_df.columns, fill_value=0)

# Predict using trained model
prediction = model.predict(user_input_processed)[0]
prediction_proba = model.predict_proba(user_input_processed)[0][1]

# Risk Assessment based on probability
risk_level = "⚠️ High Risk" if prediction_proba >= 0.5 else "✅ Low Risk"

# Display prediction clearly
st.sidebar.subheader("📝 Prediction Result")
result = "❌ Missed Task" if prediction_proba >= 0.5 else "✅ Task Likely Completed"
st.sidebar.write(f"**Prediction:** {result}")
st.sidebar.write(f"Probability of missing the task: {prediction_proba:.2%}")
st.sidebar.markdown(f"**Risk Level:** {risk_level}")

# Recommendation for User
if risk_level == "⚠️ High Risk":
    st.sidebar.warning("Consider assigning more experienced staff or increasing priority.")
else:
    st.sidebar.success("Current assignment looks good!")

