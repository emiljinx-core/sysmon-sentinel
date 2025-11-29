import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Paths
# ---------------------------------
ROOT = r"C:\SLM_Project"
RESULTS_DIR = os.path.join(ROOT, "results")
META_MODEL_PATH = os.path.join(RESULTS_DIR, "stacking_meta_model.pkl")
OUTPUT_FIG = os.path.join(RESULTS_DIR, "stacking_feature_importance.png")

# ---------------------------------
# Load saved meta-learner
# ---------------------------------
print("Loading stacking meta-learner from:", META_MODEL_PATH)

with open(META_MODEL_PATH, "rb") as f:
    meta = pickle.load(f)

coef = meta.coef_[0]  # logistic regression weights (DistilBERT, TinyBERT)

# ---------------------------------
# Plotting
# ---------------------------------
plt.figure(figsize=(6, 5))
plt.bar(["DistilBERT", "TinyBERT"], coef, color=["#4c72b0", "#c44e52"])
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Stacking Ensemble Feature Importance (Logistic Regression)", fontsize=14)
plt.ylabel("Coefficient Weight")
plt.tight_layout()

plt.savefig(OUTPUT_FIG)
plt.close()

print("Feature importance plot saved to:", OUTPUT_FIG)
print("Coefficients:", coef)
