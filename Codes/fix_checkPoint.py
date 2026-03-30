import pandas as pd
df = pd.read_csv("results_checkpoint.csv")
print(f"Önce: {len(df)} satır")
df["Email_ID"] = pd.to_numeric(df["Email_ID"], errors="coerce")
df = df[df["Email_ID"] <= 2474]
print(f"Sonra: {len(df)} satır")
df.to_csv("results_checkpoint.csv", index=False)
print("Kaydedildi!")