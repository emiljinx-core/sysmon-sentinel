import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
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

def predict_labels(model, tokenizer, texts, batch_size=32, max_len=256):
    model.eval()
    all_preds = []
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
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return acc, p, r, f1


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



# Main

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

    print("Predicting with DistilBERT...")
    distil_preds = predict_labels(distil_model, distil_tok, texts)

    print("Predicting with TinyBERT...")
    tiny_preds = predict_labels(tiny_model, tiny_tok, texts)


    base_df = pd.DataFrame({
        "true_label": y_true,
        "distilbert_pred": distil_preds,
        "tinybert_pred": tiny_preds,
    })
    base_df.to_csv(os.path.join(RESULTS_DIR, "preds_base_models.csv"), index=False)

    # OR (malicious if ANY says malicious)
    or_preds = np.where((distil_preds + tiny_preds) >= 1, 1, 0)

    # AND (malicious only if BOTH say malicious)
    and_preds = np.where((distil_preds + tiny_preds) == 2, 1, 0)

    # Benign-tie-breaker (if disagree → benign)
    benign_tie_preds = np.where(distil_preds == tiny_preds, distil_preds, 0)


    ens_df = base_df.copy()
    ens_df["hard_or"] = or_preds
    ens_df["hard_and"] = and_preds
    ens_df["hard_benign_tie"] = benign_tie_preds
    ens_df.to_csv(os.path.join(RESULTS_DIR, "preds_hard_voting_variants.csv"), index=False)


    print("\n=== Metrics on Test Set ===")
    models = [
        ("DistilBERT", distil_preds),
        ("TinyBERT", tiny_preds),
        ("Hard OR", or_preds),
        ("Hard AND", and_preds),
        ("Benign Tie", benign_tie_preds)
    ]

    for name, preds in models:
        acc, p, r, f1 = compute_metrics(y_true, preds)
        print(f"{name:15} - Acc: {acc:.4f}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
        plot_confmat(y_true, preds, f"{name} Confusion Matrix",os.path.join(RESULTS_DIR, f"confmat_{name.replace(' ','_').lower()}.png"))

    print("\nSaved predictions & confusion matrices to:", RESULTS_DIR)

if __name__ == "__main__":
    main()
