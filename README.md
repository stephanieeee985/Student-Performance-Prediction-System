# Student Performance Prediction System

## Problem
Academic risk is often noticed only after poor results. This classroom prototype
explores whether historical student information can flag learners who may need
additional academic support.

## Dataset
UCI Machine Learning Repository — Student Performance (ID 320)
https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction Target
`G3 < 10` → **AT RISK** (1); `G3 >= 10` → **NOT AT RISK** (0).
`G3` is used only to build the target and is never an input feature (prevents target leakage).

## Technologies
Python, pandas, scikit-learn (Random Forest), Streamlit, joblib, GitHub

## Installation and Run
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains model, creates student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
In the app, download the sample input CSV, then upload it to see predictions.

## Project Structure
```
student-performance-prediction/
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Limitations
- Trained on one Portuguese secondary-school dataset; may not generalize to other institutions or populations.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
- Accuracy alone is not enough; review precision, recall, and F1 for the "At risk" class.
