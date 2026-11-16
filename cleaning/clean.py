import pandas as pd

df = pd.read_csv('data/review.csv')

print(df.head())
print(df.shape)
print(df.columns)

df = df.drop_duplicates()
df = df.dropna()

df["review"] = df["review"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
df = df[df["review"] != ""]
df = df[df["sentiment"].isin(["positive", "negative"])]

df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

df.to_csv('data/cleaned_review.csv', index=False)

print("Cleaned dataset saved successfully!")