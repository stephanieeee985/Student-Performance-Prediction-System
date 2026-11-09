# train_model.py
import io
import zipfile
import urllib.request

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------
# Activity A: Retrieve and inspect the dataset
# ---------------------------------------------------------------------
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def load_math_from_zip():
    """Download the UCI zip and read the Mathematics file (student-mat.csv)."""
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))

    def find_math(z):
        for name in z.namelist():
            if name.endswith("student-mat.csv"):
                return z, name
        for name in z.namelist():  # the CSVs may be inside a nested zip
            if name.endswith(".zip"):
                inner = zipfile.ZipFile(io.BytesIO(z.read(name)))
                found = find_math(inner)
                if found:
                    return found
        return None

    found = find_math(outer)
    if found is None:
        raise FileNotFoundError("student-mat.csv not found in the UCI zip file")
    z, name = found
    return pd.read_csv(z.open(name), sep=";")


def load_from_ucimlrepo():
    """Fallback: the lab's ucimlrepo method (ID 320)."""
    from ucimlrepo import fetch_ucirepo

    student = fetch_ucirepo(id=320)
    return pd.concat([student.data.features, student.data.targets], axis=1)


try:
    df = load_math_from_zip()
    print("Loaded Mathematics file (student-mat.csv) from UCI.")
except Exception as error:
    print("Zip download failed (", error, ") - using ucimlrepo instead.")
    df = load_from_ucimlrepo()

df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# Checkpoint: G1, G2 and G3 must exist
for col in ["G1", "G2", "G3"]:
    assert col in df.columns, f"Column {col} is missing from the dataset"

# ---------------------------------------------------------------------
# Activity B: Preprocess, train, and evaluate
# ---------------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor
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

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
