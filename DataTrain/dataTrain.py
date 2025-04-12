import os
import pandas as pd

base_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_dir, "cleaned_x_hates.csv"))

print(df.head())