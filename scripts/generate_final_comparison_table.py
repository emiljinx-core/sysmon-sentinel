import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

# ---------------------------------
# Paths (FIXED)
# ---------------------------------
ROOT = r"C:\SLM_Project"
RESULTS_DIR = os.path.join(ROOT, "results")

BASE_MODELS_CSV   = os.path.join(RESULTS_DIR, "preds_base_models.csv")
HARD_VOTING_CSV   = os.path.join(RESULTS_DIR, "preds_hard_voting_variants.csv")
SOFT_VOTING_CSV   = os.path.join(RESULTS_DIR, "preds_soft_voting.csv")
STACKING_CSV      = os.path.join(RESULTS_DIR, "preds_stacking.csv")

OUTPUT_COMPARISON = os.path.join(RESULTS_DIR, "final_model_comparison.csv")

# ---------------------------------
# Metric function
# ---------------------------------
def compute_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    auc = (
        roc_auc_score(y_true, y_score)
        if y_score is not None
           and len(np.unique(y_true)) == 2
        else float("nan")
    )
    return acc, p, r, f1, auc

# ---------------------------------
# Load data (FIXED)
# ---------------------------------
print("Loading prediction CSVs...")
df_base  = pd.read_csv(BASE_MODELS_CSV)
df_hard  = pd.read_csv(HARD_VOTING_CSV)
df_soft  = pd.read_csv(SOFT_VOTING_CSV)
df_stack = pd.read_csv(STACKING_CSV)

y_true = df_base["true_label"].values

# ---------------------------------
# Collect all metrics (FIXED)
# ---------------------------------
rows = []

def add_row(name, pred, score=None):
    acc, p, r, f1, auc = compute_metrics(y_true, pred, score)
    rows.append([name, acc, p, r, f1, auc])

# Base models
add_row("DistilBERT", df_base["distilbert_pred"].values)
add_row("TinyBERT", df_base["tinybert_pred"].values)

# Hard voting variants
add_row("Hard OR Voting", df_hard["hard_or"].values)
add_row("Hard AND Voting", df_hard["hard_and"].values)
add_row("Benign Tie-Break", df_hard["hard_benign_tie"].values)

# Soft voting
add_row("Soft Voting",
        df_soft["pred_ensemble_soft"].values,
        df_soft["p_ensemble"].values)

# Stacking ensemble
add_row("Stacking Ensemble",
        df_stack["pred_stacking"].values,
        df_stack["p_stacking"].values)

# ---------------------------------
# Create table
# ---------------------------------
columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
comparison_df = pd.DataFrame(rows, columns=columns)

# Save
comparison_df.to_csv(OUTPUT_COMPARISON, index=False)

print("\n=== FINAL MODEL PERFORMANCE TABLE ===\n")
print(comparison_df)
print("\nSaved to:", OUTPUT_COMPARISON)
