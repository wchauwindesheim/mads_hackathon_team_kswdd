from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import pandas as pd
import sys 
# Load your data
train_path = sys.argv[1]
train_data = pd.read_parquet(train_path)

# Separate features and target
X = train_data.drop(columns=["target"]).values  # Features
y = train_data["target"].values  # Target (labels)

# Check class distribution before SMOTE
print("Before SMOTE:", Counter(y))
print("Original dataset size:", len(y))


# Define undersampled classes and their desired sizes
# Example: Class 1 and 3 are underrepresented, and we want 2223 and 641 samples, respectively
undersampled_classes = {1: 4446, 3: 3500}

# Apply SMOTE for only the undersampled classes
smote = SMOTE(sampling_strategy=undersampled_classes, random_state=42)

# Perform SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check class distribution after SMOTE
print("After SMOTE:", Counter(y_resampled))
print("Resampled dataset size:", len(y_resampled))

# Convert the resampled data back to DataFrame (optional)
resampled_data = pd.DataFrame(X_resampled, columns=train_data.columns[:-1])
resampled_data["target"] = y_resampled

# Calculate the total number of added samples
total_added_samples = len(y_resampled) - len(y)
print("Total added samples after SMOTE:", total_added_samples)

# Save or use the resampled data
print(resampled_data.head())
class_counts_resampled = resampled_data["target"].value_counts()

# Print total number of samples for each class
print("Total number of values for each class after SMOTE:")
print(class_counts_resampled)



valid_path = "hackathon-data/heart_big_valid.parq"
valid_data = pd.read_parquet(valid_path)

X_train = resampled_data.drop(columns=["target"]).values  # Assuming "target" is the label column
y_train = resampled_data["target"].values

X_valid = valid_data.drop(columns=["target"]).values
y_valid = valid_data["target"].values