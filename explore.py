import pandas as pd

df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "text"])
print(df.head())
print(df.shape)
print(df["label"].value_counts())