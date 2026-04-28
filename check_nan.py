import pandas as pd

df = pd.read_csv("data/processed/test_features.csv")
nan_per_row = df.head(35).isnull().sum(axis=1)
print("NaN counts for first 35 rows:")
for i, n in enumerate(nan_per_row):
    d = df["Date"].iloc[i]
    marker = " <-- " if n > 0 else ""
    print(str(i).ljust(3) + str(d) + " : " + str(n) + " NaNs" + marker)

nan_cols = [c for c in df.columns if df[c].head(35).isnull().any()]
print("\nColumns with NaN in first 35 rows: " + str(len(nan_cols)))
for c in nan_cols[:10]:
    print("  " + c + ": " + str(df[c].head(35).isnull().sum()) + " NaN rows")

print("\nTotal rows: " + str(len(df)))
print("Date range: " + str(df["Date"].min()) + " to " + str(df["Date"].max()))

# Also check last 5 rows
print("\nLast 5 rows NaN counts:")
nan_last = df.tail(5).isnull().sum(axis=1)
for i, n in zip(df.tail(5).index, nan_last):
    d = df["Date"].iloc[i]
    print(str(i).ljust(3) + str(d) + " : " + str(n) + " NaNs")
