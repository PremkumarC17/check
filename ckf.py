import pandas as pd
import numpy as np

# Load the original dataset into a pandas DataFrame
df = pd.read_csv("crop_data.csv")

# Create a copy of the DataFrame
df_copy = df.copy()

# Loop through columns in the DataFrame
df_copy = df.copy()

# Loop through columns in the DataFrame
for col in df.columns:
    if df[col].dtype == 'object':
        # For string columns, select random values from the existing values
        random_values = np.random.choice(df[col].unique(), df.shape[0])
        # Assign the random values to the copy DataFrame
        df_copy[col] = random_values
    else:
        # For non-string columns, generate random values between the minimum and maximum values of the column
        random_values = np.random.uniform(df[col].min(), df[col].max(), df.shape[0])
        # Assign the random values to the copy DataFrame
        df_copy[col] = random_values

# Concatenate the original DataFrame and the copy DataFrame
df = pd.concat([df]*100, ignore_index=True)

# Save the new dataset to a CSV file
df.to_csv("new_dataset.csv", index=False)