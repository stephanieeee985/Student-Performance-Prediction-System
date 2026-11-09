# Model Evaluation Results

Random Forest (100 trees, class_weight="balanced"), 80/20 stratified split, random_state=42.
Target: 1 = AT RISK (G3 < 10), 0 = NOT AT RISK. G3 excluded from the inputs.

Run `python train_model.py`, then copy the numbers from the terminal output
(or from the generated `evaluation_results.json`) into this table.

| Metric | Value |
| --- | --- |
| Test set size | ___ |
| Accuracy | ___ |
| Precision (At risk) | ___ |
| Recall (At risk) | ___ |
| F1-score (At risk) | ___ |

Confusion matrix (test set): TN = ___, FP = ___, FN = ___, TP = ___

**Interpretation (2–3 sentences):**
Recall for the "At risk" class is the share of truly at-risk students the model catches; a missed
at-risk student (false negative) is the costlier error here. Precision shows how many flagged
students were truly at risk. Do not judge the model on accuracy alone.
