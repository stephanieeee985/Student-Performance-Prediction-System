# Student Performance Prediction System

## Problem
Academic risk is often noticed only after poor results. This classroom prototype
estimates whether a student is at risk of failing the final Mathematics grade,
using historical student data.

## Dataset
UCI Machine Learning Repository, Student Performance (ID 320):
https://archive.ics.uci.edu/dataset/320/student+performance
(Mathematics file, `student-mat.csv`)

## Prediction target
`AT RISK` if G3 < 10, `NOT AT RISK` if G3 >= 10. G3 is the target only and is
never used as an input feature (to avoid target leakage).

## Technologies
Python, pandas, scikit-learn (Random Forest), joblib, Streamlit, ucimlrepo, GitHub

## Installation and run
```
pip install -r requirements.txt
python train_model.py     # downloads data, trains, saves student_model.pkl
streamlit run app.py      # opens at http://localhost:8501
```
In the app, download the sample input CSV, then upload it to see predictions.

## Limitations
- Trained on one Portuguese secondary-school dataset; may not generalize to
  other institutions or populations.
- Predictions are not diagnoses and must not be the sole basis for real
  decisions about students.
