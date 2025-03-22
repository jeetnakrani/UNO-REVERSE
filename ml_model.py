
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_predictive_model(df):
    df_model = df.copy()
    features = ['expected_duration', 'technician_experience', 'asset_age', 
                'prev_misses_by_person', 'prev_misses_by_asset', 'priority']
    df_model = pd.get_dummies(df_model[features + ['was_missed']], drop_first=True)

    X = df_model.drop('was_missed', axis=1)
    y = df_model['was_missed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)

    return model, report
