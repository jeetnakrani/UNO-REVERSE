import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Title
st.title("🚀 Smart Task Management Dashboard")

# Load Data
@st.cache
def load_data():
    data = pd.read_csv("facility_tasks.csv")  # Your CSV file path
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Custom Filters")
teams = st.sidebar.multiselect("Select Team", options=data["team"].unique(), default=data["team"].unique())
tasks = st.sidebar.multiselect("Select Task Type", options=data["task_type"].unique(), default=data["task_type"].unique())
dates = st.sidebar.date_input("Select Date Range", [data["date"].min(), data["date"].max()])

filtered_data = data[
    (data['team'].isin(teams)) &
    (data['task_type'].isin(tasks)) &
    (data['date'] >= pd.to_datetime(dates[0])) &
    (data['date'] <= pd.to_datetime(dates[1]))
]

# Prediction Model
X = pd.get_dummies(filtered_data[['task_type', 'team', 'task_duration']])
y = filtered_data['task_missed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("📊 Prediction Performance")
st.text(classification_report(y_test, y_pred))

# SHAP Explanation
st.subheader("📌 SHAP Feature Importance")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
st.pyplot(bbox_inches='tight')

# Heatmap Visualization
st.subheader("🔥 Missed Tasks Heatmap")
heatmap_data = filtered_data.pivot_table(values='task_missed', index='team', columns='task_type', aggfunc='sum', fill_value=0)
fig1 = px.imshow(heatmap_data, color_continuous_scale="Reds")
st.plotly_chart(fig1)

# Automated Insights
st.subheader("🤖 Automated Insights")
total_missed = filtered_data['task_missed'].sum()
top_team = filtered_data.groupby('team')['task_missed'].sum().idxmax()
top_task = filtered_data.groupby('task_type')['task_missed'].sum().idxmax()

st.info(f"Total missed tasks: {total_missed}")
st.warning(f"🚩 Team '{top_team}' has the highest missed tasks.")
st.warning(f"⚠️ Task '{top_task}' type misses most frequently.")
st.success(f"✅ Recommendation: Prioritize resources for '{top_team}' on '{top_task}' tasks.")

# Prophet Forecasting
forecast_df = filtered_data.groupby('date').agg({'task_missed':'sum'}).reset_index().rename(columns={'date':'ds','task_missed':'y'})
prophet = Prophet()
prophet.fit(forecast_df)
future = prophet.make_future_dataframe(30)
forecast = prophet.predict(future)

st.subheader("📅 30-Day Forecast")
fig2 = prophet.plot(forecast)
st.pyplot(fig2)

# Anomaly Detection
iso = IsolationForest(contamination=0.05, random_state=42)
filtered_data['anomaly'] = iso.fit_predict(X)
anomalies = filtered_data[filtered_data['anomaly']==-1]

st.subheader("🚨 Detected Anomalies")
st.write(anomalies[['date', 'team', 'task_type', 'task_duration']])

# Resource Optimization
daily_avg = filtered_data.groupby('date').size().mean()
optimal_staff = np.ceil(daily_avg / 10)
st.subheader("👥 Resource Optimization")
st.info(f"Optimal daily staffing recommendation: {int(optimal_staff)} members.")

# Adaptive Scheduling Alerts
latest_day = filtered_data['date'].max()
latest_miss_rate = filtered_data[filtered_data['date']==latest_day]['task_missed'].mean()

if latest_miss_rate > filtered_data['task_missed'].mean():
    st.error("🔴 Alert: Increased missed tasks detected today. Reallocation recommended.")
else:
    st.success("🟢 Today's missed tasks are within expected ranges.")

# K-Means Clustering
st.subheader("🧩 Task Clustering")
kmeans = KMeans(n_clusters=3, random_state=42)
filtered_data['cluster'] = kmeans.fit_predict(X)

fig3 = px.scatter(filtered_data, x="task_duration", y="task_type", color="cluster", title="Task Clusters (Completion Likelihood)")
st.plotly_chart(fig3)
