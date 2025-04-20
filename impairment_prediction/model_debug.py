from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

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
        
        # Use all sequence outputs instead of just the last one
        output = output.mean(dim=1)  # Average over sequence length
        
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output  # Return raw logits

# 1) Build a Dataset that yields (seq_tensor, label_int)
label_map = {'no': 0, 'yes': 1}

class DrivingDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # Print class distribution
        print("\nClass distribution:")
        class_dist = df.groupby('id')['impaired'].first().value_counts(normalize=True)
        print(class_dist)
        
        self.data = []
        
        for run_id, run_data in df.groupby('id'):
            run_data = run_data.sort_values('time_from_start')
            
            # Get coordinates
            coords = run_data[['x', 'y']].values
            
            # Normalize coordinates
            # For x: divide by max x value to get relative position
            # For y: divide by road width (12) to get lane position
            coords[:, 0] = coords[:, 0] / coords[:, 0].max()
            coords[:, 1] = coords[:, 1] / 6
            
            # Ensure time steps are uniform
            time_steps = run_data['time_from_start'].values
            expected_time_steps = np.arange(0, len(time_steps) * 1/30, 1/30)
            if not np.allclose(time_steps, expected_time_steps, atol=1e-6):
                print(f"Warning: Non-uniform time steps detected for run {run_id}")
            
            coords = torch.tensor(coords, dtype=torch.float)
            impaired_flag = run_data['impaired'].iloc[0]
            label = label_map[impaired_flag]
            
            self.data.append((coords, label))
            
            # Print first few coordinates for debugging
            if run_id == 1:
                print("\nFirst run coordinates (normalized):")
                print(coords[:5])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 2) Split into train/val/test
full_ds = DrivingDataset("simulation_data.csv")
n = len(full_ds)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val
train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test],
                                       generator=torch.Generator().manual_seed(42))

# 3) A collate_fn that pads and also converts labels to tensor
def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.float)
    return padded, labels, lengths

# 4) DataLoaders
batch_size = 32
train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                        collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                      collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size, shuffle=False,
                       collate_fn=collate_fn)

# Initialize model
model = GRUModel(2, hidden_size=64, num_layers=2, dropout=0.3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Use weighted BCE loss to handle class imbalance
pos_weight = torch.tensor([2.0]).to(device)  # Adjust this based on your class distribution
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

def evaluate(model, loader, criterion, return_probs=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities
    
    with torch.no_grad():
        for padded, labels, lengths in loader:
            padded, labels, lengths = padded.to(device), labels.to(device), lengths.to(device)
            logits = model(padded, lengths)
            loss = criterion(logits.squeeze(), labels)
            total_loss += loss.item() * labels.size(0)
            
            # Get probabilities using sigmoid
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            
            # For classification metrics, still use 0.5 threshold
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    if return_probs:
        return avg_loss, cm, report, np.array(all_probs)
    return avg_loss, cm, report

# Training loop
num_epochs = 50
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    
    for padded, labels, lengths in train_loader:
        padded, labels, lengths = padded.to(device), labels.to(device), lengths.to(device)
        logits = model(padded, lengths)
        loss = criterion(logits.squeeze(), labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
    
    avg_train_loss = total_loss / len(train_loader.dataset)
    
    # Evaluate on validation set
    val_loss, val_cm, val_report = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)
    
    print(f"\nEpoch {epoch}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print("\nValidation Confusion Matrix:")
    print(val_cm)
    print("\nValidation Classification Report:")
    print(val_report)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# Load best model and evaluate on test set
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_cm, test_report, test_probs = evaluate(model, test_loader, criterion, return_probs=True)
print("\nFinal Test Results:")
print(f"Test Loss: {test_loss:.4f}")
print("\nTest Confusion Matrix:")
print(test_cm)
print("\nTest Classification Report:")
print(test_report)

# Print some example probabilities
print("\nExample probabilities for first 10 test samples:")
for i, prob in enumerate(test_probs[:10]):
    print(f"Sample {i+1}: Probability of impairment = {prob[0]:.4f}")

    