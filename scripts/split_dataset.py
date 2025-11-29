import pandas as pd
from sklearn.model_selection import train_test_split

input_path = r"C:\SLM_Project\logs\cleaned_events.csv"
output_dir = r"C:\SLM_Project\logs"


df = pd.read_csv(input_path)


train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label"]
)


val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)


train_df.to_csv(f"{output_dir}/train.csv", index=False)
val_df.to_csv(f"{output_dir}/val.csv", index=False)
test_df.to_csv(f"{output_dir}/test.csv", index=False)

print("Split completed!")
print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

print("\nLabel distribution in Train:")
print(train_df['label'].value_counts())

print("\nLabel distribution in Validation:")
print(val_df['label'].value_counts())

print("\nLabel distribution in Test:")
print(test_df['label'].value_counts())
