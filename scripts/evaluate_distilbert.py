import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = r"C:\SLM_Project\models\distilbert-sysmon-final"
TEST_CSV = r"C:\SLM_Project\logs\test.csv"

print("Loading tokenizer and model...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Loading test data")
df = pd.read_csv(TEST_CSV)

texts = df["Message"].tolist()
labels = df["label"].tolist()

print("Tokenizing test samples...")
encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
encodings = {k: v.to(device) for k, v in encodings.items()}

with torch.no_grad():
    outputs = model(**encodings)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

print("\n--- DISTILBERT CLASSIFICATION REPORT ---")
print(classification_report(labels, preds, digits=4))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("DistilBERT Confusion Matrix")
plt.savefig(r"C:\SLM_Project\results\confmat_distilbert.png")
plt.show()
