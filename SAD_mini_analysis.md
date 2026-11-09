# SAD Mini-Analysis — Student Performance Prediction System

| SAD element | Response |
|---|---|
| Problem statement | Academic risk is often noticed after poor results. The prototype explores early identification of students at risk of failing the final Mathematics grade using available student indicators. |
| Users | Faculty member or academic analyst. |
| Inputs | Student attributes and academic indicators: study time, absences, failures, G1, G2, and other demographic/social columns. |
| Process | Validate data (required columns) → preprocess (impute, scale, encode) → apply trained Random Forest model → classify risk. |
| Outputs | AT RISK / NOT AT RISK label per student, downloadable results CSV, and model evaluation summary. |
| Key limitation | A model trained on one Portuguese secondary-school dataset may not generalize to another institution or population. |

**Mini use-case:** The user uploads a CSV → the system checks required columns → the model predicts risk → the system displays and exports the results.
