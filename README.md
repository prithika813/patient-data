import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data (replace 'data.csv')
df = pd.read_csv('data.csv').dropna()
X = df[['age', 'gender_male', 'symptom1', 'symptom2']] # Adjust features
y = df['diagnosis']

# Preprocess categorical (example)
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

def predict(data):
    df_new = pd.DataFrame([data])
    df_new = pd.get_dummies(df_new, columns=['gender'], drop_first=True).reindex(columns=X.columns, fill_value=0)
    return model.predict(df_new)[0]

# Example prediction
new_patient = {'age': 55, 'gender': 'female', 'symptom1': 1, 'symptom2': 0}
print(f"Prediction: {predict(new_patient)}")# patient-data
