
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    df_anomaly = df.copy()
    daily_counts = df_anomaly.groupby('DATE')['was_missed'].sum().reset_index()

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    daily_counts['anomaly'] = iso_forest.fit_predict(daily_counts[['was_missed']])

    anomalies = daily_counts[daily_counts['anomaly'] == -1]
    return anomalies
