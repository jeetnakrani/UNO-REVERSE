import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("task_data.csv")

# Convert target column to binary
df["status"] = df["status"].map({"Done": 0, "Missed": 1})

# Select features and target
features = ['task_type', 'team_name', 'duration (mins)', 'workload', 'resource_available', 'day_of_week']
X = df[features]
y = df["status"]

# Encode categorical features
label_encoders = {}
for col in ['task_type', 'team_name', 'day_of_week']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n✅ Model Training Completed!\n")
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n💾 Model saved as model.pkl")
print("💾 Encoders saved as label_encoders.pkl")
