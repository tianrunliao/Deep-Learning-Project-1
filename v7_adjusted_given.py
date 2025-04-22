import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import struct
import pandas as pd

# Load MNIST dataset
def load_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Data paths
train_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# Load and preprocess data
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

train_images = torch.tensor(train_images, dtype=torch.float32) / 255.0
test_images = torch.tensor(test_images, dtype=torch.float32) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

train_images = train_images.view(-1, 28 * 28)
test_images = test_images.view(-1, 28 * 28)

# No need for one-hot encoding with cross-entropy loss
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define MLP model with configurable nHidden and Dropout
class BasicMLP(nn.Module):
    def __init__(self, nHidden, dropout_rate=0.0):
        super(BasicMLP, self).__init__()
        layers = []
        input_dim = 28 * 28
        for hidden_units in nHidden:
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.Sigmoid())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_units
        layers.append(nn.Linear(input_dim, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)  # Outputs raw logits

# Training function with regularization and early stopping
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, regularization=None, l1_lambda=0.0, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)  # Labels are class indices
            if regularization == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Validation for early stopping
        if regularization == 'early_stopping' and val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

# Testing function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Define nHidden candidates
nHidden_candidates = [
    [128],          # One hidden layer with 128 neurons
    [256, 128],     # Two hidden layers with 256 and 128 neurons
    [512, 256, 128] # Three hidden layers with 512, 256, and 128 neurons
]

# Train final model with fixed hyperparameters and compare regularizations
def train_final_model():
    print("\nTraining final model with fixed hyperparameters...")
    torch.manual_seed(42)

    # Fixed hyperparameters
    lr = 0.01
    num_epochs = 10
    batch_size = 64
    nHidden_index = 1  # Choose [256, 128]
    momentum = 0.9
    dropout_rate = 0.3
    l1_lambda = 0.0001
    weight_decay = 0.0001
    nHidden = nHidden_candidates[nHidden_index]

    print(f"Selected nHidden configuration: {nHidden}")
    print(f"Selected momentum: {momentum}")
    print(f"Selected dropout_rate: {dropout_rate}")
    print(f"Selected l1_lambda: {l1_lambda}")
    print(f"Selected weight_decay: {weight_decay}")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(train_images, train_labels)  # Using train as val for simplicity
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    regularization_types = ['none', 'l1', 'l2', 'dropout', 'early_stopping']
    results = {}

    for reg in regularization_types:
        print(f"\nRunning final model with {reg} regularization")
        if reg == 'dropout':
            model = BasicMLP(nHidden, dropout_rate=dropout_rate)
        else:
            model = BasicMLP(nHidden)
        
        criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss
        if reg == 'l2':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        train(model, train_loader, val_loader if reg == 'early_stopping' else None, 
              criterion, optimizer, num_epochs, regularization=reg, l1_lambda=l1_lambda if reg == 'l1' else 0.0)
        
        test_accuracy = test(model, test_loader)
        results[reg] = test_accuracy
        print(f'Final Test Accuracy with {reg}: {test_accuracy:.2f}%')

    # Create and print results table
    df = pd.DataFrame(list(results.items()), columns=['Regularization', 'Test Accuracy (%)'])
    print("\n=== Final Results Table ===")
    print(df)

# Train the final model with fixed hyperparameters
train_final_model()