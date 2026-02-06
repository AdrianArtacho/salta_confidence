import pandas as pd
import matplotlib.pyplot as plt

# load your file
df = pd.read_csv("OUTPUT/exp5/modality_agreement_timeseries.csv")

plt.figure(figsize=(10, 4))

for col in df.columns:
    if col != "time":
        plt.plot(df["time"], df[col], label=col, alpha=0.8)

plt.xlabel("Time (s)")
plt.ylabel("Pairwise agreement")
plt.title("Time-resolved modality agreement")
plt.ylim(0, 1)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

