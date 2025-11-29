import pandas as pd

PRED_PATH = r"C:\SLM_Project\results\preds_base_models.csv"
df = pd.read_csv(PRED_PATH)

# Identify disagreement cases
disagree = df[df["distilbert_pred"] != df["tinybert_pred"]]

print("Total disagreements:", len(disagree))
print("\n--- Disagreement Breakdown ---")

# DistilBERT=0 TinyBERT=1  → TinyBERT caught malicious, DistilBERT missed
type1 = disagree[(disagree["distilbert_pred"] == 0) & (disagree["tinybert_pred"] == 1)]

# DistilBERT=1 TinyBERT=0 → DistilBERT false positive
type2 = disagree[(disagree["distilbert_pred"] == 1) & (disagree["tinybert_pred"] == 0)]

print("Type 1 (Distil=0, Tiny=1):", len(type1))
print("Type 2 (Distil=1, Tiny=0):", len(type2))

# Save results
type1.to_csv(r"C:\SLM_Project\results\disagree_type1_distil_missed.csv", index=False)
type2.to_csv(r"C:\SLM_Project\results\disagree_type2_distil_fp.csv", index=False)

print("\nSaved:")
print(" - disagree_type1_distil_missed.csv (DistilBERT miss; TinyBERT correct)")
print(" - disagree_type2_distil_fp.csv (DistilBERT false positive)")
