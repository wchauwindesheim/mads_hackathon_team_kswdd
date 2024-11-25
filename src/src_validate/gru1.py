import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

# Load the Parquet files
train_path = "hackathon-data/heart_big_train.parq"
valid_path = "hackathon-data/heart_big_valid.parq"

# Load data into Pandas DataFrames
train_data = pd.read_parquet(train_path)
valid_data = pd.read_parquet(valid_path)

# Separate features and targets
X_train = train_data.drop(columns=["target"]).values  # Assuming "target" is the label column
y_train = train_data["target"].values

X_valid = valid_data.drop(columns=["target"]).values
y_valid = valid_data["target"].values

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Add input_size dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(-1)  # Add input_size dimension
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define the GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        """
        GRU-based model for classification.
        Args:
            input_size (int): Number of features in the input sequence.
            hidden_size (int): Number of GRU hidden units.
            output_size (int): Number of output classes.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout rate for regularization.
        """
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the GRU model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # GRU output
        out, _ = self.gru(x, h0)  # out: (batch_size, seq_length, hidden_size)

        # Use the output from the last time step
        out = self.fc(out[:, -1, :])  # out: (batch_size, output_size)
        return out

if __name__ == "__main__":
    # Model parameters
    input_size = 1  # Univariate time series (1 feature)
    hidden_size = 128 # Number of hidden units in GRU
    output_size = 5  # Number of output classes
    num_layers = 3  # Number of GRU layers
    dropout = 0.5  # Dropout rate

    # Instantiate the model
    model = GRUModel(input_size, hidden_size, output_size, num_layers, dropout)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Print classification report
    print(classification_report(y_true, y_pred, target_names=["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]))
    model_save_path = "gru_model.pth"  # Path to save the model
    torch.save(model, model_save_path)
    print(f"Model saved at {model_save_path}")