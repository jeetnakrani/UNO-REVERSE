import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import base64
import io

st.set_page_config(page_title="Smart Facility Task Dashboard", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("facility_tasks.csv", parse_dates=['scheduled_time', 'start_time', 'completion_time'])

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

df = load_data()
model = load_model()

# Preprocessing
df['hour_of_day'] = df['scheduled_time'].dt.hour
df['weekday'] = df['scheduled_time'].dt.strftime('%A')
df['month'] = df['scheduled_time'].dt.month
if 'was_missed' not in df.columns:
    df['was_missed'] = df['status'].apply(lambda x: 1 if str(x).strip().lower() == 'missed' else 0)

# Sidebar: Filter and Prediction
st.sidebar.title("🛠️ Controls")

# Filters
st.sidebar.header("📂 Filter Dataset")
task_types = st.sidebar.multiselect("Task Types", df['task_type'].unique(), default=list(df['task_type'].unique()))
locations = st.sidebar.multiselect("Locations", df['location'].unique(), default=list(df['location'].unique()))
filtered_df = df[(df['task_type'].isin(task_types)) & (df['location'].isin(locations))]

# Prediction inputs
st.sidebar.header("🔮 Predict Missed Task")
expected_duration = st.sidebar.slider("Expected Duration (min)", 30, 240, 60)
technician_experience = st.sidebar.slider("Technician Experience (yrs)", 1, 15, 5)
hour_of_day = st.sidebar.slider("Scheduled Hour", 0, 23, 10)
asset_age = st.sidebar.slider("Asset Age (yrs)", 0, 20, 5)
priority = st.sidebar.selectbox("Priority", ['Low', 'Medium', 'High'])
priority_enc = {'Low': 0, 'Medium': 1, 'High': 2}[priority]

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([{
        'expected_duration': expected_duration,
        'technician_experience': technician_experience,
        'hour_of_day': hour_of_day,
        'asset_age': asset_age,
        'priority_enc': priority_enc
    }])
    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    risk = "High Risk" if prob > 0.7 else ("Medium Risk" if prob > 0.4 else "Low Risk")
    status = "Missed" if prediction == 1 else "Completed"
    st.sidebar.markdown(f"### 🔍 Prediction: **{status}** ({risk}, Confidence: {prob:.2%})")

# Main Panel
st.title("📊 Smart Facility Task Dashboard")
st.markdown("Track task performance, detect patterns, and predict missed tasks using ML.")

# Show filtered data
st.subheader("📄 Filtered Dataset")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# Heatmap
st.subheader("🔥 Miss Rate by Weekday & Hour")
heatmap_data = filtered_df.pivot_table(index='weekday', columns='hour_of_day', values='was_missed', aggfunc='mean')
fig_heat = px.imshow(heatmap_data, color_continuous_scale='Reds', aspect='auto')
st.plotly_chart(fig_heat, use_container_width=True)

# Trend
st.subheader("📈 Monthly Miss Rate Trend")
trend = filtered_df.groupby('month')['was_missed'].mean().reset_index()
fig_trend = px.line(trend, x='month', y='was_missed', markers=True)
st.plotly_chart(fig_trend, use_container_width=True)

# Quick Insight
st.subheader("🧠 Quick Insight")
latest_month = trend['month'].max()
miss_rate = trend.loc[trend['month'] == latest_month, 'was_missed'].values[0]
st.markdown(f"**Latest Month ({latest_month}) Miss Rate:** {miss_rate:.2%}")

# SHAP Explainability
st.subheader("🔍 Feature Importance (SHAP)")
important_features = ['expected_duration', 'technician_experience', 'hour_of_day', 'asset_age', 'priority']
df_model = df.copy()
df_model['priority_enc'] = df_model['priority'].map({'Low': 0, 'Medium': 1, 'High': 2})
df_model = df_model.dropna(subset=important_features)
X_shap = df_model[['expected_duration', 'technician_experience', 'hour_of_day', 'asset_age', 'priority_enc']]

explainer = shap.Explainer(model, X_shap)
shap_values = explainer(X_shap.iloc[:100], check_additivity=False)
fig = plt.figure()
shap.summary_plot(shap_values, X_shap.iloc[:100], show=False)
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format="png")
plt.close(fig)
buf.seek(0)
image_base64 = base64.b64encode(buf.read()).decode('utf-8')
st.image(f"data:image/png;base64,{image_base64}", caption="Top Contributing Features")
