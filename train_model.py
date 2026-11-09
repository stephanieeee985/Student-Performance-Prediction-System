# train_model.py
# Student Performance Prediction System - data retrieval, training, evaluation
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------------
# Activity A - Retrieve and inspect the dataset
# ---------------------------------------------------------------------------
# Fetch UCI Student Performance dataset (ID 320)
student = fetch_ucirepo(id=320)

# Combine student features and grade targets (G1, G2, G3)
df = pd.concat([student.data.features, student.data.targets], axis=1)

# Save the records as a CSV file
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# ---------------------------------------------------------------------------
# Activity B - Preprocess, train, and evaluate
# ---------------------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor (prevents target leakage)
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy, 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
))

# Save the headline metrics for the "At risk" class (the class we care about)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, predictions, labels=[1], zero_division=0
)
tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
metrics = {
    "test_size": int(len(y_test)),
    "accuracy": round(float(accuracy), 3),
    "at_risk_precision": round(float(precision[0]), 3),
    "at_risk_recall": round(float(recall[0]), 3),
    "at_risk_f1": round(float(f1[0]), 3),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}
with open("evaluation_results.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved evaluation_results.json:", metrics)

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
