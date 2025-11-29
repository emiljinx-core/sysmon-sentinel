import pandas as pd

input_path = r"C:\SLM_Project\logs\merged_events.csv"
clean_path = r"C:\SLM_Project\logs\cleaned_events.csv"

df = pd.read_csv(input_path)

df = df.dropna(subset=['Message'])
df = df[df['Message'].astype(str).str.strip() != ""]


df = df.drop_duplicates(subset=['Message'])

df['Message'] = df['Message'].astype(str)
df['Message'] = df['Message'].str.replace(r'\s+', ' ', regex=True)
df['Message'] = df['Message'].str.strip()

df = df[['Message', 'label']]

df.to_csv(clean_path, index=False)

print("\nCleaned dataset saved to:", clean_path)
print("Dataset size:", len(df))
print(df['label'].value_counts())
