# Student Performance Prediction System

A classroom prototype that estimates whether a student is at risk of failing
the final Mathematics grade, built for the Data Mining laboratory exercise.

## Problem
Academic risk is often noticed only after poor results. This prototype explores
early identification of students who may need extra academic support, using
indicators such as study time, absences, past failures, and the period grades
G1 and G2.

## Dataset
UCI Machine Learning Repository, *Student Performance* (ID 320):
https://archive.ics.uci.edu/dataset/320/student+performance
Public, historical records from Portuguese secondary schools.

## Prediction target
`G3 < 10` is **AT RISK** (1); `G3 >= 10` is **NOT AT RISK** (0).
`G3` is used only to build the target and is never an input feature.

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest pipeline), joblib, Streamlit,
ucimlrepo, Git/GitHub.

## Installation and run
```bash
git clone https://github.com/YOUR-USERNAME/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt

python train_model.py     # downloads data, trains, evaluates, saves student_model.pkl
streamlit run app.py      # opens http://localhost:8501
```
In the app, download the sample input CSV, then upload it to see predictions.
Uploading a CSV with missing predictor columns shows an error listing them.

## Model evaluation
Run `python train_model.py` and record the printed accuracy, precision,
recall and F1-score here:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Not at risk | _fill in_ | _fill in_ | _fill in_ |
| At risk | _fill in_ | _fill in_ | _fill in_ |

Accuracy: _fill in_ (80/20 stratified split, `random_state=42`)

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to
  other institutions or populations.
- G1 and G2 are strong predictors, so the model is less useful for *early*
  warning, since those grades exist only partway through the course.
- Predictions are not diagnoses and must not be the sole basis for real
  student decisions. This is a classroom demonstration.
