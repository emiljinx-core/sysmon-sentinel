import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

test_path = r"C:/SLM_Project/logs/test.csv"
model_path = r"C:/SLM_Project/models/tinybert-sysmon-final"

df_test = pd.read_csv(test_path)

texts = df_test["Message"].tolist()
labels = df_test["label"].tolist()

tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")   # IMPORTANT
model = BertForSequenceClassification.from_pretrained(model_path)       # Your fine-tuned weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)
encodings = {k: v.to(device) for k, v in encodings.items()}

with torch.no_grad():
    outputs = model(**encodings)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

print("\n--- TINYBERT CLASSIFICATION REPORT ---\n")
print(classification_report(labels, preds, digits=4))

cm = confusion_matrix(labels, preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("TinyBERT Confusion Matrix")

save_path = r"C:/SLM_Project/results/confmat_tinybert.png"
plt.savefig(save_path)

plt.show()

print(f"\nConfusion matrix saved to: {save_path}")
