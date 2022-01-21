from matplotlib import pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

exp_df_path = "../assets/v5_1.csv"

df = pd.read_csv(exp_df_path)
print(df.columns)

df = df[["CV (raw)", "CV (with pp)", "Public LB"]]

cleaned_df = df[df["Public LB"].notnull()]

print(cleaned_df)

plt.figure()
plt.title("Public LB vs CV (with pp)")
plt.xlabel("LB")
plt.ylabel("CV (with pp)")
# plt.scatter(cleaned_df["Public LB"], cleaned_df["CV (raw)"])
plt.scatter(cleaned_df["Public LB"], cleaned_df["CV (with pp)"])

plt.show()
