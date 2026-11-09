# SAD Mini-Analysis — Student Performance Prediction System

| SAD element | Response |
| --- | --- |
| **Problem statement** | Academic risk is often noticed only after poor results are recorded. This prototype explores early identification of students at risk of failing the final grade (G3 < 10) using indicators that are already available, such as study time, past failures, absences, and the period grades G1 and G2. |
| **Users** | Faculty member or academic analyst who reviews groups of students and decides where extra academic support may be needed. |
| **Inputs** | A CSV of student records with the same predictor columns as the training data: demographic and social attributes (e.g. age, school, family background), study habits (study time, failures, absences), and earlier grades (G1, G2). G3 is never an input. |
| **Process** | Validate uploaded data (required columns present) → preprocess (impute missing values, scale numeric columns, one-hot encode categorical columns) → apply the trained Random Forest model → classify each record as at risk or not at risk. |
| **Outputs** | An AT RISK / NOT AT RISK label for every uploaded record (viewable and downloadable as CSV), a count of predicted at-risk records, and a model evaluation summary (accuracy, precision, recall, F1-score). |
| **Key limitation** | The model is trained on one Portuguese secondary-school dataset, so it may not generalize to another institution, curriculum, or population. Predictions are not diagnoses and must not be the sole basis for real decisions about students. |

**Mini use-case:** The user uploads a CSV file → the system checks required columns → the model predicts risk → the system displays and exports the results.
