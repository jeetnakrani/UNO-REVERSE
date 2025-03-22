import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# Load the preprocessed dataset
df = pd.read_csv("processed_facility_data.csv")

# Separate features and target
X = df.drop(columns=["was_missed"])
y = df["was_missed"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate
for name, model in models.items():
    print(f"\n🔹 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred))

    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"📈 ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Optional: Save best model (e.g., Random Forest) to file
import joblib
joblib.dump(models["Random Forest"], "random_forest_model.pkl")
print("\n🧠 Random Forest model saved to 'random_forest_model.pkl'")
