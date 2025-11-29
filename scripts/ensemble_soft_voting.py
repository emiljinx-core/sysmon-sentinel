import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    BertTokenizerFast,
    BertForSequenceClassification
)

ROOT = r"C:\SLM_Project"

TEST_CSV = os.path.join(ROOT, "logs", "test.csv")

DISTIL_MODEL_DIR = os.path.join(ROOT, "models", "distilbert-sysmon-final")
TINY_MODEL_DIR   = os.path.join(ROOT, "models", "tinybert-sysmon-final")

RESULTS_DIR  = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_probs(model, tokenizer, texts, batch_size=32, max_len=256):
    """Return probability of class 1 (malicious)."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


def compute_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    auc = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    return acc, p, r, f1, auc


def plot_confmat(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("Loading test set...")
    df = pd.read_csv(TEST_CSV)


    texts = df["Message"].tolist()
    y_true = df["label"].values


    print("Loading DistilBERT...")
    distil_tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    distil_model = DistilBertForSequenceClassification.from_pretrained(
        DISTIL_MODEL_DIR
    ).to(device)

    print("Loading TinyBERT...")
    tiny_tok = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
    tiny_model = BertForSequenceClassification.from_pretrained(
        TINY_MODEL_DIR
    ).to(device)


    print("Getting DistilBERT probabilities...")
    p_distil = predict_probs(distil_model, distil_tok, texts)

    print("Getting TinyBERT probabilities...")
    p_tiny = predict_probs(tiny_model, tiny_tok, texts)

    # Soft Voting
    p_ens = (p_distil + p_tiny) / 2.0
    y_pred_ens = (p_ens >= 0.5).astype(int)


    out_df = pd.DataFrame({
        "true_label": y_true,
        "p_distil": p_distil,
        "p_tiny": p_tiny,
        "p_ensemble": p_ens,
        "pred_ensemble_soft": y_pred_ens,
    })
    out_path = os.path.join(RESULTS_DIR, "preds_soft_voting.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved soft-voting predictions to: {out_path}")


    print("\n=== SOFT VOTING METRICS ===")
    acc_e, p_e, r_e, f1_e, auc_e = compute_metrics(y_true, y_pred_ens, p_ens)

    print(f"Soft Voting  - Acc: {acc_e:.4f}, P: {p_e:.4f}, R: {r_e:.4f}, F1: {f1_e:.4f}, AUC: {auc_e:.4f}")

    confmat_path = os.path.join(RESULTS_DIR, "confmat_soft_voting.png")
    plot_confmat(y_true, y_pred_ens, "Soft Voting Ensemble Confusion Matrix", confmat_path)

    print(f"\nSaved ensemble confusion matrix to: {confmat_path}")
    print("Soft voting ensemble complete.")


if __name__ == "__main__":
    main()
