import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ---------------------------------
# Paths
# ---------------------------------
ROOT = r"C:\SLM_Project"
RESULTS_DIR = os.path.join(ROOT, "results")

BASE_MODELS_CSV   = os.path.join(RESULTS_DIR, "preds_base_models.csv")
SOFT_VOTING_CSV   = os.path.join(RESULTS_DIR, "preds_soft_voting.csv")
STACKING_CSV      = os.path.join(RESULTS_DIR, "preds_stacking.csv")

OUTPUT_FIG = os.path.join(RESULTS_DIR, "roc_all_models.png")

# ---------------------------------
# Load prediction data
# ---------------------------------
df_base  = pd.read_csv(BASE_MODELS_CSV)
df_soft  = pd.read_csv(SOFT_VOTING_CSV)
df_stack = pd.read_csv(STACKING_CSV)

y_true = df_base["true_label"].values

# Probabilities
p_distil = df_base["distil_prob"].values if "distil_prob" in df_base else None
p_tiny   = df_base["tiny_prob"].values if "tiny_prob" in df_base else None

# If base CSV does not have prob columns, load soft/stack
if p_distil is None:
    print("Distil prob not found in base csv — using soft_voting csv")
    p_distil = df_soft["p_distil"].values

if p_tiny is None:
    print("Tiny prob not found in base csv — using soft_voting csv")
    p_tiny = df_soft["p_tiny"].values

p_soft  = df_soft["p_ensemble"].values
p_stack = df_stack["p_stacking"].values

# ---------------------------------
# Compute ROC curves
# ---------------------------------
def get_roc(y, scores):
    fpr, tpr, _ = roc_curve(y, scores)
    auc_value = auc(fpr, tpr)
    return fpr, tpr, auc_value

roc_curves = {
    "DistilBERT": get_roc(y_true, p_distil),
    "TinyBERT": get_roc(y_true, p_tiny),
    "Soft Voting": get_roc(y_true, p_soft),
    "Stacking": get_roc(y_true, p_stack),
}

# ---------------------------------
# Plot ROC
# ---------------------------------
plt.figure(figsize=(8, 6))

for name, (fpr, tpr, auc_value) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.4f})", linewidth=2)

plt.plot([0, 1], [0, 1], "k--", label="Random Guess")

plt.title("ROC Curve Comparison — DistilBERT, TinyBERT, Ensembles", fontsize=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_FIG)
plt.close()

print("ROC curves saved to:", OUTPUT_FIG)
