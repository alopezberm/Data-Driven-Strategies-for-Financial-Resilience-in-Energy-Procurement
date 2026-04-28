import pandas as pd

df = pd.read_csv("data/processed/modeling_dataset.csv")
print("Shape:", df.shape)
print("Date col:", [c for c in df.columns if "date" in c.lower()][:3])
print("First cols:", df.columns[:5].tolist())
df["date"] = pd.to_datetime(df["date"])
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print("Total rows:", len(df))
# Check NaN in first row of 2025
test = df[df["date"] >= "2025-01-01"].copy()
print("2025 rows:", len(test))
if len(test) > 0:
    first_row_nans = test.head(30).isnull().sum(axis=1)
    print("NaN counts for first 30 rows of 2025:")
    for i, n in enumerate(first_row_nans):
        d = test["date"].iloc[i].date()
        marker = " <-- NaN" if n > 0 else ""
        print("  " + str(d) + ": " + str(n) + marker)
