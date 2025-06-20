import pandas as pd
df = pd.read_csv("Grade vs Admissions Inflation.csv")
correlation = df['Grade Inflation z-scores'].corr(df['UCI Admissions z-scores'])
print(f"Correlation: {correlation:.4f}")
