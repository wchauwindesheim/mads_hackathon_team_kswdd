import argparse
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mads_hackathon_team_kswdd.src.src_validate.borderline import GRUModel
from sklearn.metrics import classification_report

def main():
    # Argument parser for configuration
    parser = argparse.ArgumentParser(description="Evaluate a GRU model on the validation dataset.")
    parser.add_argument('--train_path', type=str, required=True, help="Path to the training dataset (Parquet format).")
    parser.add_argument('--valid_path', type=str, required=True, help="Path to the validation dataset (Parquet format).")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved GRU model (.pth).")
    parser.add_argument('--output_file', type=str, default='predictions_with_probabilities_gru_model.csv', 
                        help="Path to save the output predictions with probabilities.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument('--hidden_size', type=int, default=128, help="Number of hidden units in GRU.")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of GRU layers.")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for GRU.")
    parser.add_argument('--output_size', type=int, default=5, help="Number of output classes.")

    args = parser.parse_args()

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUModel(input_size=1, hidden_size=args.hidden_size, output_size=args.output_size, 
                     num_layers=args.num_layers, dropout=args.dropout)
    model.to(device)

    # Load model state_dict
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load data into Pandas DataFrames
    train_data = pd.read_parquet(args.train_path)
    valid_data = pd.read_parquet(args.valid_path)

    # Separate features and targets
    X_train = train_data.drop(columns=["target"]).values
    y_train = train_data["target"].values
    X_valid = valid_data.drop(columns=["target"]).values
    y_valid = valid_data["target"].values

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Perform evaluation and collect predictions
    y_true = []
    y_pred_probs = []

    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Get the model's predictions
            outputs = model(X_batch)
            # Get class probabilities using softmax
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(outputs)
            y_true.extend(y_batch.cpu().numpy())
            y_pred_probs.extend(probs.cpu().numpy())

    # Convert the prediction probabilities into a DataFrame
    prediction_data = pd.DataFrame(y_pred_probs, columns=[f'pred_class_{i}' for i in range(args.output_size)])
    prediction_data['target'] = y_true  # Add the true labels

    # Save the DataFrame to a CSV file
    prediction_data.to_csv(args.output_file, index=False)
    print(f"Predictions with probabilities saved to {args.output_file}")

if __name__ == "__main__":
    main()
