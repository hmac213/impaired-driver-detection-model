import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        output = output.mean(dim=1)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

def preprocess_trajectory(trajectory_df):
    """
    Preprocess a trajectory DataFrame to match training format.
    
    Args:
        trajectory_df: DataFrame with columns ['x', 'y', 'time_from_start']
    
    Returns:
        preprocessed tensor and length
    """
    # Sort by time
    trajectory_df = trajectory_df.sort_values('time_from_start')
    
    # Get coordinates
    coords = trajectory_df[['x', 'y']].values
    
    # Normalize coordinates
    coords[:, 0] = coords[:, 0] / coords[:, 0].max()
    coords[:, 1] = coords[:, 1] / 6  # Assuming road width of 6
    
    # Convert to tensor
    coords = torch.tensor(coords, dtype=torch.float)
    
    return coords, len(coords)

def predict_impairment(model, trajectory_df, device='cpu'):
    """
    Predict impairment probability for a given trajectory.
    
    Args:
        model: Trained GRUModel
        trajectory_df: DataFrame with columns ['x', 'y', 'time_from_start']
        device: Device to run model on ('cpu' or 'cuda')
    
    Returns:
        probability of impairment (float between 0 and 1)
    """
    # Preprocess trajectory
    coords, length = preprocess_trajectory(trajectory_df)
    
    # Prepare batch
    coords = coords.unsqueeze(0)  # Add batch dimension
    lengths = torch.tensor([length], dtype=torch.long)
    
    # Move to device
    coords = coords.to(device)
    lengths = lengths.to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(coords, lengths)
        probability = torch.sigmoid(logits).item()
    
    return probability

def main():
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUModel(2, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Example usage
    # Load your trajectory data
    trajectory_df = pd.read_csv('path_to_your_trajectory.csv')
    
    # Get prediction
    probability = predict_impairment(model, trajectory_df, device)
    
    # Print results
    print(f"Probability of impairment: {probability:.4f}")
    print(f"Classification: {'Impaired' if probability > 0.5 else 'Not Impaired'}")
    
    # You can also set custom thresholds
    if probability > 0.7:
        print("High confidence of impairment - Immediate action recommended")
    elif probability > 0.5:
        print("Possible impairment - Monitor closely")
    else:
        print("No significant signs of impairment")

if __name__ == "__main__":
    main() 