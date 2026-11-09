# Submission – Student Performance Prediction System

## 1. SAD mini-analysis
| Element | Response |
|---|---|
| Problem statement | Academic risk is often noticed only after poor results. This prototype explores early identification of students at risk of failing final Mathematics using available student indicators. |
| User | Faculty member or academic analyst. |
| Inputs | Student attributes and academic indicators (study time, absences, past failures, G1, G2, etc.) via a CSV upload. |
| Process | Validate required columns → preprocess (impute, scale, encode) → apply trained Random Forest → classify risk. |
| Outputs | AT RISK / NOT AT RISK label per record, a count of predicted at-risk records, a downloadable results CSV, and the model evaluation summary. |
| Key limitation | Trained on one Portuguese secondary-school dataset; it may not generalize to another institution or population. |

## 2. Code files
`train_model.py`, `app.py` (see project folder).

## 3. Supporting files
`requirements.txt`, `README.md` (see project folder).

## 4. Model evaluation results
Copy these from your own run of `python train_model.py`:

- Accuracy: ______
- Precision (At risk): ______  Recall (At risk): ______  F1 (At risk): ______
- Precision (Not at risk): ______  Recall (Not at risk): ______  F1 (Not at risk): ______

## 5. Screenshots
Insert: (a) the dashboard at http://localhost:8501, (b) the prediction results after uploading `sample_prediction_input.csv`.

## 6. GitHub repository URL
https://github.com/YOUR-USERNAME/student-performance-prediction

---

## Testing (section 10)
| Test | Expected result | Pass? |
|---|---|---|
| Run train_model.py | CSV and model files created without errors | ☐ |
| Run Streamlit app | Dashboard loads in browser | ☐ |
| Upload sample CSV | Predictions displayed | ☐ |
| Upload CSV missing a required column | Helpful error message | ☐ |
| Download results | CSV with predictions downloaded | ☐ |
| Open GitHub repository | Source files and README visible | ☐ |

## Reflection answers
1. **Target and G3 exclusion.** The target is whether the final Mathematics grade G3 is below 10 (1 = at risk, 0 = not at risk). G3 is excluded from the features because it defines the target; including it would be target leakage, giving unrealistically high scores and a model that cannot work on students whose final grade isn't known yet.
2. **Recall for "At risk".** It is the share of students who truly failed (G3 < 10) that the model correctly flagged. Low recall means many struggling students are missed, which matters most here because a missed at-risk student gets no support.
3. **Different performance at my institution.** The data come from two Portuguese schools in a specific period. Grading scales, curriculum, demographics, school policies and the distribution of features can differ, so the learned patterns may not transfer.
4. **Action after an AT RISK prediction.** Treat it as a prompt for a conversation, not a verdict: review the student's actual record, talk with the student and relevant teachers or advisers, and offer support such as tutoring or counseling. Never use the prediction as the sole basis for a decision.
