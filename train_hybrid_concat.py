import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
import argparse

# Import the hybrid model from quantum_model.py
from quantum_model import HybridConcatModel

# -------------------------------
# Dataset class
# -------------------------------
class EnhancedDataset(Dataset):
    def __init__(self, mobilevit_dir, freq_dir):
        self.mobilevit_dir = mobilevit_dir
        self.freq_dir = freq_dir
        self.files = []
        self.labels = []
        for fname in os.listdir(mobilevit_dir):
            if fname.endswith('.npy'):
                base = fname.replace('.npy', '')
                freq_path = os.path.join(freq_dir, base + '.pt')
                if os.path.exists(freq_path):
                    self.files.append(base)
                    self.labels.append(0 if 'real' in base else 1)
        print(f"EnhancedDataset: {len(self.files)} samples ({self.labels.count(0)} real, {self.labels.count(1)} fake)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        base = self.files[idx]
        vit_feat = torch.from_numpy(np.load(os.path.join(self.mobilevit_dir, base + '.npy'))).float()
        freq_feat = torch.load(os.path.join(self.freq_dir, base + '.pt'))
        # If freq_feat has 4 dimensions (from earlier version), take first two
        if freq_feat.shape[0] == 4:
            freq_feat = freq_feat[:2]
        combined = torch.cat([vit_feat, freq_feat])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return combined, label

# -------------------------------
# Training function
# -------------------------------
def train(args):
    # Hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    reduced_dim = args.reduced_dim
    n_layers = args.n_layers
    hidden = args.hidden

    # Dataset
    dataset = EnhancedDataset(args.mobilevit_dir, args.freq_dir)
    if len(dataset) == 0:
        print("No samples found. Check paths.")
        return

    # Split train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Balanced sampler for training
    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[0]
    model = HybridConcatModel(input_dim=input_dim,
                              reduced_dim=reduced_dim,
                              n_layers=n_layers,
                              hidden=hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, labels in loop:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    # Save model
    os.makedirs(args.weights_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.weights_dir, args.weights_file))
    print(f"Training complete. Weights saved to {os.path.join(args.weights_dir, args.weights_file)}")

# -------------------------------
# Command-line interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train hybrid quantum-classical deepfake detector.')
    parser.add_argument('--mobilevit_dir', type=str, default='mobilevit_features',
                        help='Directory with .npy MobileNetV3 features.')
    parser.add_argument('--freq_dir', type=str, default='precomputed_features_subset',
                        help='Directory with .pt frequency features.')
    parser.add_argument('--weights_dir', type=str, default='weights',
                        help='Directory to save trained weights.')
    parser.add_argument('--weights_file', type=str, default='hybrid_concat_weights.pth',
                        help='Filename for saved weights.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--reduced_dim', type=int, default=4, help='Dimensionality after reducer.')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of quantum layers.')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden size in classical networks.')
    args = parser.parse_args()

    train(args)
