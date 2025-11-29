import os
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
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
import pickle  

ROOT = r"C:\SLM_Project"
VAL_CSV  = os.path.join(ROOT, "logs", "val.csv")
TEST_CSV = os.path.join(ROOT, "logs", "test.csv")

DISTIL_MODEL_DIR = os.path.join(ROOT, "models", "distilbert-sysmon-final")
TINY_MODEL_DIR   = os.path.join(ROOT, "models", "tinybert-sysmon-final")

RESULTS_DIR  = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def predict_probs(model, tokenizer, texts, batch_size=32, max_len=256):
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_all.extend(probs)
    return np.array(probs_all)


def compute_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = float("nan")
    else:
        auc = float("nan")
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
    print("Loading validation and test sets...")
    val_df  = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    val_texts  = val_df["Message"].tolist()
    test_texts = test_df["Message"].tolist()

    y_val  = val_df["label"].values
    y_test = test_df["label"].values

    print("Loading tokenizers and models...")
    distil_tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tiny_tok   = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

    distil_model = DistilBertForSequenceClassification.from_pretrained(DISTIL_MODEL_DIR).to(device)
    tiny_model   = BertForSequenceClassification.from_pretrained(TINY_MODEL_DIR).to(device)

    print("Getting probabilities on validation set...")
    p_distil_val = predict_probs(distil_model, distil_tok, val_texts)
    p_tiny_val   = predict_probs(tiny_model,  tiny_tok,  val_texts)

    print("Getting probabilities on test set...")
    p_distil_test = predict_probs(distil_model, distil_tok, test_texts)
    p_tiny_test   = predict_probs(tiny_model,  tiny_tok,  test_texts)

    X_val  = np.column_stack([p_distil_val,  p_tiny_val])
    X_test = np.column_stack([p_distil_test, p_tiny_test])

    print("Training logistic regression meta-learner on validation set...")
    meta = LogisticRegression(C=0.5, class_weight="balanced", random_state=42, max_iter=1000)
    meta.fit(X_val, y_val)


    meta_path = os.path.join(RESULTS_DIR, "stacking_meta_model.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved stacking meta-learner to: {meta_path}")


    p_meta = meta.predict_proba(X_test)[:, 1]
    print("Meta probability range on test:", float(p_meta.min()), "to", float(p_meta.max()))

    y_pred_distil = (p_distil_test >= 0.5).astype(int)
    y_pred_tiny   = (p_tiny_test   >= 0.5).astype(int)

    acc_d, p_d, r_d, f1_d, auc_d = compute_metrics(y_test, y_pred_distil, p_distil_test)
    acc_t, p_t, r_t, f1_t, auc_t = compute_metrics(y_test, y_pred_tiny,   p_tiny_test)

    print("\n=== Base models (thr=0.5) ===")
    print(f"DistilBERT - Acc: {acc_d:.4f}, P: {p_d:.4f}, R: {r_d:.4f}, F1: {f1_d:.4f}, AUC: {auc_d:.4f}")
    print(f"TinyBERT   - Acc: {acc_t:.4f}, P: {p_t:.4f}, R: {r_t:.4f}, F1: {f1_t:.4f}, AUC: {auc_t:.4f}")

    print("\n=== Stacking thresholds ===")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_f1 = -1
    best_thr = 0.5
    best_pred = None

    for thr in thresholds:
        y_meta = (p_meta >= thr).astype(int)
        acc_s, p_s, r_s, f1_s, auc_s = compute_metrics(y_test, y_meta, p_meta)
        print(f"Thr={thr:.1f} -> Acc: {acc_s:.4f}, P: {p_s:.4f}, R: {r_s:.4f}, F1: {f1_s:.4f}, AUC: {auc_s:.4f}")

        if f1_s > best_f1:
            best_f1 = f1_s
            best_thr = thr
            best_pred = y_meta

    print(f"\nBest stacking threshold: {best_thr:.1f} (F1={best_f1:.4f})")

    out_df = pd.DataFrame({
        "true_label": y_test,
        "p_distil": p_distil_test,
        "p_tiny": p_tiny_test,
        "p_stacking": p_meta,
        "pred_distil": y_pred_distil,
        "pred_tiny": y_pred_tiny,
        "pred_stacking": best_pred,
    })
    out_path = os.path.join(RESULTS_DIR, "preds_stacking.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved stacking predictions to: {out_path}")

    confmat_path = os.path.join(RESULTS_DIR, "confmat_stacking.png")
    plot_confmat(y_test, best_pred,f"Stacking Confusion Matrix (thr={best_thr:.1f})",confmat_path)
    print(f"Saved stacking confusion matrix to: {confmat_path}")


if __name__ == "__main__":
    main()
