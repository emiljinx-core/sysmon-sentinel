import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# ---------------------------------
# Paths
# ---------------------------------
ROOT = r"C:\SLM_Project"
RESULTS_DIR = os.path.join(ROOT, "results")

BASE_MODELS_CSV   = os.path.join(RESULTS_DIR, "preds_base_models.csv")
SOFT_VOTING_CSV   = os.path.join(RESULTS_DIR, "preds_soft_voting.csv")
STACKING_CSV      = os.path.join(RESULTS_DIR, "preds_stacking.csv")

OUTPUT_FIG = os.path.join(RESULTS_DIR, "pr_all_models.png")

# ---------------------------------
# Load prediction data
# ---------------------------------
df_base  = pd.read_csv(BASE_MODELS_CSV)
df_soft  = pd.read_csv(SOFT_VOTING_CSV)
df_stack = pd.read_csv(STACKING_CSV)

y_true = df_base["true_label"].values

# Probability extraction
p_distil = df_soft["p_distil"].values
p_tiny   = df_soft["p_tiny"].values
p_soft   = df_soft["p_ensemble"].values
p_stack  = df_stack["p_stacking"].values

# ---------------------------------
# Compute PR curves
# ---------------------------------
def get_pr(y, scores):
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    return precision, recall, ap

pr_curves = {
    "DistilBERT": get_pr(y_true, p_distil),
    "TinyBERT": get_pr(y_true, p_tiny),
    "Soft Voting": get_pr(y_true, p_soft),
    "Stacking": get_pr(y_true, p_stack),
}

# ---------------------------------
# Plot PR curves
# ---------------------------------
plt.figure(figsize=(8, 6))

for name, (prec, rec, ap) in pr_curves.items():
    plt.plot(rec, prec, label=f"{name} (AP = {ap:.4f})", linewidth=2)

plt.title("Precision–Recall Curve Comparison", fontsize=14)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend(loc="lower left")
plt.tight_layout()

plt.savefig(OUTPUT_FIG)
plt.close()

print("PR curves saved to:", OUTPUT_FIG)
