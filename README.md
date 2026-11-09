# Student Performance Prediction System

A classroom prototype that predicts whether a student is **at risk of failing the final grade (G3)**
from historical student records. Built for the Data Mining laboratory exercise.

> **Responsible use:** This is a demonstration using public, historical data. Predictions are not
> diagnoses and must not be used as the sole basis for real student decisions.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification
using available student indicators (study time, absences, failures, G1, G2, and others) so that a
faculty member or academic analyst can see which students may need additional support.

## Dataset
UCI Machine Learning Repository — [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
(ID 320), downloaded automatically with `ucimlrepo` and saved as `student_data.csv`.

## Prediction target
`G3 < 10` → **AT RISK** (1); `G3 >= 10` → **NOT AT RISK** (0).
G3 is the target only and is never used as an input feature (to avoid target leakage).

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), Streamlit, ucimlrepo, joblib, GitHub.

## Installation and run steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the data, train and evaluate the model
#    (creates student_data.csv, student_model.pkl, evaluation_results.json)
python train_model.py

# 3. Launch the app (normally http://localhost:8501)
streamlit run app.py
```
In the app, download the sample input CSV, upload it, and review/download the predictions.
Uploading a CSV with a missing required column shows an error message listing the missing columns.

## Files
| File | Purpose |
| --- | --- |
| `train_model.py` | Downloads the dataset, preprocesses, trains and evaluates the model |
| `app.py` | Streamlit interface: dataset overview, CSV upload, predictions, download |
| `requirements.txt` | Python dependencies |
| `SAD_analysis.md` | SAD mini-analysis (problem, user, inputs, process, outputs, limitation) |
| `evaluation_results.md` | Accuracy, precision, recall and F1-score of the trained model |

## Limitations
- Trained on a single Portuguese secondary-school dataset; it may not generalize to other
  institutions or populations.
- Small dataset, so metrics vary with the train/test split.
- Uses G1 and G2, so it flags risk only after those grades exist; it is not a very early warning.
- Accuracy alone is misleading; check recall and precision for the "At risk" class.
