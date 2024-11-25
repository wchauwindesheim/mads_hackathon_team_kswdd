import pandas as pd
import numpy as np

# Configuration
num_rows = 100  # Number of rows in the CSV
output_file = "predictions1.csv"  # Output CSV file

# Generate random prediction probabilities for 5 classes
predictions = np.random.rand(num_rows, 5)  # Random values for pred_class_0 to pred_class_4
predictions = predictions / predictions.sum(axis=1, keepdims=True)  # Normalize to sum to 1

# Generate random target classes (e.g., 0 to 4)
targets = np.random.randint(0, 5, size=(num_rows, 1))

# Combine predictions and targets into a DataFrame
columns = [f"pred_class_{i}" for i in range(5)] + ["target"]
data = np.hstack([predictions, targets])
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)
print(f"CSV file '{output_file}' with {num_rows} rows created successfully!")
