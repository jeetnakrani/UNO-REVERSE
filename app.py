import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import timedelta

# Set page config
st.set_page_config(page_title="Smart Task Dashboard", layout="wide")

# Load model & encoders
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load data
@st.cache_data(ttl=5)
def load_data():
    try:
        df = pd.read_csv("task_data.csv", parse_dates=["date"])
        return df
    except FileNotFoundError:
        st.error("task_data.csv not found.")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.warning("No data available.")
    st.stop()

# Features used in training
training_features = [
    'task_type',
    'team_name',
    'duration (mins)',
    'workload',
    'resource_available',
    'day_of_week'
]

# ---------------------- SIDEBAR ---------------------- #

st.sidebar.header("🔎 Filter Tasks")
team = st.sidebar.multiselect("Select Team(s)", df["team_name"].unique(), default=df["team_name"].unique())
status = st.sidebar.multiselect("Select Status", df["status"].unique(), default=df["status"].unique())
date_range = st.sidebar.date_input("Select Date Range", [])

# ML toggle
st.sidebar.markdown("---")
predict_on = st.sidebar.checkbox("🧠 Enable ML Predictions", value=True)

# Custom task predictor
st.sidebar.markdown("### 🧪 Predict a New Task")

with st.sidebar.form("predict_form"):
    task_type_input = st.selectbox("Task Type", df["task_type"].unique())
    team_input = st.selectbox("Team", df["team_name"].unique())
    duration_input = st.slider("Duration (mins)", 10, 60, 30)
    workload_input = st.slider("Workload (1-10)", 1, 10, 5)
    resource_input = st.slider("Resource Availability", 0.3, 1.0, 0.7, step=0.01)
    day_input = st.selectbox("Day of Week", df["day_of_week"].unique())

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    input_df = pd.DataFrame([{
        "task_type": task_type_input,
        "team_name": team_input,
        "duration (mins)": duration_input,
        "workload": workload_input,
        "resource_available": resource_input,
        "day_of_week": day_input
    }])

    try:
        # Encode categorical fields
        for col in ['task_type', 'team_name', 'day_of_week']:
            input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        result = "❌ Missed" if prediction == 1 else "✅ Done"
        st.sidebar.success(f"Prediction: **{result}**")

        # Debug
        st.sidebar.write("🧪 Encoded Model Input:")
        st.sidebar.dataframe(input_df)
        st.sidebar.write("Raw Prediction:", prediction)

    except Exception as e:
        st.sidebar.error("Error during prediction:")
        st.sidebar.write(e)

# ---------------------- FILTER MAIN DATA ---------------------- #

df = df[df["team_name"].isin(team)]
df = df[df["status"].isin(status)]
if len(date_range) == 2:
    df = df[(df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))]

# Prediction on full dataset
def predict_status(df):
    df_predict = df[training_features].copy()
    for col in ['task_type', 'team_name', 'day_of_week']:
        df_predict[col] = label_encoders[col].transform(df_predict[col])
    preds = model.predict(df_predict)
    return ["Missed" if p == 1 else "Done" for p in preds]

# ---------------------- MAIN DASHBOARD ---------------------- #

st.title("📊 Smart Task Monitoring Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tasks", len(df))
col2.metric("Completed", (df["status"] == "Done").sum())
col3.metric("Missed", (df["status"] == "Missed").sum())
rate = (df["status"].value_counts(normalize=True).get("Done", 0) * 100) if len(df) > 0 else 0
col4.metric("Completion Rate", f"{rate:.2f}%" if rate else "N/A")

# Table
if predict_on:
    df["Predicted_Status"] = predict_status(df)

    def highlight_missed(row):
        return ['background-color: #ffcccc' if row.get("Predicted_Status") == "Missed" else '' for _ in row]

    st.subheader("📋 Task Data with Predictions")
    styled_df = df.sort_values(by="date", ascending=False)[[
        'task_id', 'task_type', 'team_name', 'date', 'start_time',
        'status', 'Predicted_Status', 'workload', 'resource_available'
    ]].style.apply(highlight_missed, axis=1)
    st.dataframe(styled_df, use_container_width=True)

else:
    st.subheader("📋 Task Data")
    st.dataframe(df.sort_values(by="date", ascending=False), use_container_width=True)

# ---------------------- CHARTS ---------------------- #

st.subheader("📈 Visual Insights")

# Line Chart
st.markdown("### 🗓️ Tasks Over Last 60 Days")
end_date = df["date"].max()
start_date = end_date - timedelta(days=59)
date_range = pd.date_range(start=start_date, end=end_date)

tasks_per_day = df[df["date"] >= start_date].groupby("date")["task_id"].count().reindex(date_range, fill_value=0)
tasks_per_day.index.name = "Date"

fig1, ax1 = plt.subplots()
tasks_per_day.plot(kind="line", marker='o', ax=ax1)
ax1.set_title("Tasks per Day")
ax1.set_xlabel("Date")
ax1.set_ylabel("Number of Tasks")
ax1.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# Pie Chart
st.markdown("### 📌 Task Status Distribution")
fig2, ax2 = plt.subplots()
df["status"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax2)
ax2.set_ylabel("")
st.pyplot(fig2)

# Bar Chart
st.markdown("### 🧑‍🤝‍🧑 Tasks per Team")
fig3, ax3 = plt.subplots()
df.groupby("team_name")["task_id"].count().plot(kind="bar", ax=ax3)
ax3.set_title("Tasks by Team")
ax3.set_xlabel("Team")
ax3.set_ylabel("Task Count")
st.pyplot(fig3)

# Refresh
if st.button("🔄 Refresh Dashboard"):
    st.rerun()
