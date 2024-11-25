import pandas as pd
import torch
import argparse
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
from gru1 import GRUModel

# Function to handle SMOTE oversampling if needed
def apply_smote(X, y, undersampled_classes, random_state=42):
    smote = SMOTE(sampling_strategy=undersampled_classes, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Class distribution after SMOTE:", Counter(y_resampled))
    return X_resampled, y_resampled

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GRU model with SMOTE and class balancing.")
    parser.add_argument("train_path", type=str, help="Path to the training data.")
    parser.add_argument("valid_path", type=str, help="Path to the validation data.")
    parser.add_argument("--undersampled_classes", type=dict, default={}, 
                        help="Dictionary defining undersampled classes with desired sample sizes.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    
    return parser.parse_args()

# Main evaluation function
def main():
    # Parse the command line arguments
    args = parse_args()
    
    # Load training and validation data
    train_data = pd.read_parquet(args.train_path)
    valid_data = pd.read_parquet(args.valid_path)
    
    # Separate features and targets
    X_train = train_data.drop(columns=["target"]).values
    y_train = train_data["target"].values
    X_valid = valid_data.drop(columns=["target"]).values
    y_valid = valid_data["target"].values
    
    # If undersampled classes are specified, apply SMOTE
    if args.undersampled_classes:
        print("Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train, args.undersampled_classes, args.random_state)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(-1)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    # Define the model
    input_size = 1  # Univariate time series (1 feature)
    hidden_size = 128
    num_layers = 3
    dropout = 0.5
    output_size = 5  # Number of output classes

    # Initialize the model
    model = GRUModel(input_size, hidden_size, output_size, num_layers, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the trained model weights
    model_path = "gru_model.pth"  # Change this if the path is different
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()  # Set the model to evaluation mode

    # Evaluation
    y_true = []
    y_pred_probs = []
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            # Get class probabilities using softmax
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(outputs)
            
            y_true.extend(y_batch.cpu().numpy())
            y_pred_probs.extend(probs.cpu().numpy())
    
    # Convert the prediction probabilities into a DataFrame
    prediction_data = pd.DataFrame(y_pred_probs, columns=[f'pred_class_{i}' for i in range(output_size)])
    prediction_data['target'] = y_true  # Add the true labels

    # Save the predictions to a CSV file
    output_file = "predictions_with_probabilities_gru_model.csv"
    prediction_data.to_csv(output_file, index=False)
    print(f"Predictions with probabilities saved to '{output_file}'")

    # Optionally, print the classification report
    print(classification_report(y_true, [np.argmax(x) for x in y_pred_probs]))

if __name__ == "__main__":
    main()
