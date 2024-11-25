import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import sys
sys.path.append("/home/william/mads-hackathon")

#valid_path = "../../hackathon-data/heart_big_valid.parq"
valid_path = "/home/william/hackathon-data/heart_big_valid.parq"
valid_data = pd.read_parquet(valid_path)

# Step 2: Load the model
model_path = "transformer_kj_smote.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.load(model_path, map_location=device)

model.eval()  # Set the model to evaluation mode

##
#train_path = "../../hackathon-data/heart_big_train_smote.parq"
valid_path = "/home/william/hackathon-data/heart_big_valid.parq"
# Load data into Pandas DataFrames
#train_data = pd.read_parquet(train_path)
valid_data = pd.read_parquet(valid_path)
# Separate features and targets
#X_train = train_data.drop(columns=["target"]).values  # Assuming "target" is the label column
#y_train = train_data["target"].values
X_valid = valid_data.drop(columns=["target"]).values
y_valid = valid_data["target"].values
# Normalize the features using StandardScaler
scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_valid = scaler.transform(X_valid)
# Convert data to PyTorch tensors
#X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Add input_size dimension
#y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(-1)  # Add input_size dimension
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
# Create TensorDataset and DataLoader
#train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
#train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
##

output_size = 5


y_true = []
y_pred_probs = []
with torch.no_grad():
    for X_batch, y_batch in valid_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        # Get class probabilities using softmax
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(outputs)  # Probability distribution over classes
        y_true.extend(y_batch.cpu().numpy())
        y_pred_probs.extend(probs.cpu().numpy())
# Convert the prediction probabilities into a DataFrame
prediction_data = pd.DataFrame(y_pred_probs, columns=[f'pred_class_{i}' for i in range(output_size)])
prediction_data['target'] = y_true  # Add the true labels
# Save the DataFrame to a CSV file
prediction_data.to_csv('predictions_with_probabilities_transformer_smote_KJK.csv', index=False)


