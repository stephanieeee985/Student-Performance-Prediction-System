# train_model.py
import io
import os
import urllib.request
import zipfile

import pandas as pd

# ---------------------------------------------------------------
# PART 1: Retrieve the dataset (UCI Student Performance, ID 320)
# The dataset has two files: Mathematics (395 rows) and Portuguese
# (649 rows). This lab needs the MATHEMATICS file.
# ---------------------------------------------------------------
MATH_ROWS = 395
ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def read_math_from_zip(zf):
    """Find student-mat.csv in a zip (also looks inside a nested zip)."""
    for name in zf.namelist():
        if name.lower().endswith("student-mat.csv"):
            with zf.open(name) as f:
                return pd.read_csv(f, sep=";")
    for name in zf.namelist():
        if name.lower().endswith(".zip"):
            with zf.open(name) as f:
                inner = zipfile.ZipFile(io.BytesIO(f.read()))
            result = read_math_from_zip(inner)
            if result is not None:
                return result
    return None


def get_math_data():
    # 1) Reuse a saved copy if it is the Mathematics file
    if os.path.exists("student_data.csv"):
        saved = pd.read_csv("student_data.csv")
        if len(saved) == MATH_ROWS:
            return saved
        print(f"Existing student_data.csv has {len(saved)} rows, not "
              f"{MATH_ROWS}; downloading the Mathematics file again.")

    # 2) A student-mat.csv placed manually in this folder
    if os.path.exists("student-mat.csv"):
        return pd.read_csv("student-mat.csv", sep=";")

    # 3) Download the official zip and take student-mat.csv
    try:
        with urllib.request.urlopen(ZIP_URL, timeout=60) as resp:
            data = read_math_from_zip(zipfile.ZipFile(io.BytesIO(resp.read())))
        if data is not None:
            return data
    except Exception as exc:
        print("Zip download failed:", exc)

    # 4) Fall back to the ucimlrepo interface used in the handout
    from ucimlrepo import fetch_ucirepo
    student = fetch_ucirepo(id=320)
    return pd.concat([student.data.features, student.data.targets], axis=1)


df = get_math_data()
if len(df) != MATH_ROWS:
    raise SystemExit(
        f"Expected the Mathematics file ({MATH_ROWS} rows) but got {len(df)} rows. "
        "Download student-mat.csv from the UCI page, put it in this folder, "
        "and run the script again."
    )
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# Checkpoint: G1, G2 and G3 must all be present
for col in ["G1", "G2", "G3"]:
    assert col in df.columns, f"Column {col} is missing from the dataset"

# ---------------------------------------------------------------
# PART 2: Preprocess, train and evaluate
# ---------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0
))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
