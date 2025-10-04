import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/sharpr/processed_sharpr_sequence_activity.csv")
df['log2_ratio'].hist(bins=80)
plt.xlabel("log2(RNA/DNA)")
plt.title("SHARPR sample activity distribution")
plt.show()
