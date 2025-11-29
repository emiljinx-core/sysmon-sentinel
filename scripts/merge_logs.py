import pandas as pd

benign_path = r"C:\SLM_Project\logs\benign_events.csv"
malicious_path = r"C:\SLM_Project\logs\malicious_events.csv"
output_path = r"C:\SLM_Project\logs\merged_events.csv"

df_benign = pd.read_csv(benign_path)
df_benign['label'] = 0

df_malicious = pd.read_csv(malicious_path)
df_malicious['label'] = 1

df = pd.concat([df_benign, df_malicious], ignore_index=True)

print("Total samples:", len(df))
print(df['label'].value_counts())

df.to_csv(output_path, index=False)
print("Merged dataset saved at:", output_path)
