import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from scipy.ndimage import shift, rotate, zoom

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

# Data augmentation function
def augment_data(images):
    augmented_images = []
    
    for image in images:
        # Original image
        augmented_images.append(image)
        
        # Translation: Random shift by -2 to 2 pixels
        shift_x, shift_y = np.random.randint(-2, 3, size=2)
        shifted_image = shift(image, [shift_x, shift_y], mode='nearest')
        augmented_images.append(shifted_image)
        
        # Rotation: Random rotation between -10 and 10 degrees
        angle = np.random.uniform(-10, 10)
        rotated_image = rotate(image, angle, reshape=False, mode='nearest')
        augmented_images.append(rotated_image)
        
        # Scaling: Random zoom between 0.9 and 1.1
        scale = np.random.uniform(0.9, 1.1)
        scaled_image = zoom(image, scale, mode='nearest')
        # Ensure image remains 28x28
        if scaled_image.shape != (28, 28):
            scaled_image = scaled_image[:28, :28] if scaled_image.shape[0] > 28 else np.pad(
                scaled_image, ((0, 28-scaled_image.shape[0]), (0, 28-scaled_image.shape[1])), mode='constant')
        augmented_images.append(scaled_image)
    
    return np.array(augmented_images)

# Data paths
train_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# Load data
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# Apply data augmentation to training images
print("Applying data augmentation...")
train_images_augmented = augment_data(train_images)
# Augmentation creates 4x images (original + 3 augmentations per image)
train_labels_augmented = np.repeat(train_labels, 4, axis=0)

# Convert to tensors and preprocess
train_images_augmented = torch.tensor(train_images_augmented, dtype=torch.float32) / 255.0
test_images = torch.tensor(test_images, dtype=torch.float32) / 255.0
train_labels_augmented = torch.tensor(train_labels_augmented, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Flatten images
train_images_augmented = train_images_augmented.view(-1, 28 * 28)
test_images = test_images.view(-1, 28 * 28)

# One-hot encode labels
def to_one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

train_labels_one_hot = to_one_hot(train_labels_augmented)
test_labels_one_hot = to_one_hot(test_labels)

# Create test data loader
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels_one_hot)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define MLP model
class BasicMLP(nn.Module):
    def __init__(self, nHidden):
        super(BasicMLP, self).__init__()
        layers = []
        input_dim = 28 * 28
        for hidden_units in nHidden:
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.Sigmoid())
            input_dim = hidden_units
        layers.append(nn.Linear(input_dim, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# Testing function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels, 1)
            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Define nHidden candidates
nHidden_candidates = [
    [128],          # One hidden layer with 128 neurons
    [256, 128],     # Two hidden layers with 256 and 128 neurons
    [512, 256, 128] # Three hidden layers with 512, 256, and 128 neurons
]

# Objective function for Bayesian optimization
def objective(lr, num_epochs, batch_size, k_folds, nHidden_index):
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    nHidden_index = int(nHidden_index)
    nHidden = nHidden_candidates[nHidden_index]

    torch.manual_seed(42)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images_augmented)):
        print(f'Fold {fold+1}/{k_folds}')
        train_images_fold = train_images_augmented[train_idx]
        train_labels_fold = train_labels_one_hot[train_idx]
        val_images_fold = train_images_augmented[val_idx]
        val_labels_fold = train_labels_one_hot[val_idx]

        train_dataset_fold = torch.utils.data.TensorDataset(train_images_fold, train_labels_fold)
        val_dataset_fold = torch.utils.data.TensorDataset(val_images_fold, val_labels_fold)

        train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        val_loader_fold = torch.utils.data.DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

        model = BasicMLP(nHidden)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        train(model, train_loader_fold, criterion, optimizer, num_epochs)
        val_accuracy = test(model, val_loader_fold)
        fold_accuracies.append(val_accuracy)
        print(f'Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')

    avg_val_accuracy = np.mean(fold_accuracies)
    print(f'Average Validation Accuracy: {avg_val_accuracy:.2f}%')
    return avg_val_accuracy

# Set up Bayesian optimization
pbounds = {
    'lr': (0.001, 0.1),
    'num_epochs': (5, 15),
    'batch_size': (16, 128),
    'k_folds': (3, 5),
    'nHidden_index': (0, len(nHidden_candidates) - 1)
}

optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

# Run Bayesian optimization
print("Starting Bayesian Optimization...")
optimizer.maximize(init_points=5, n_iter=10)

# Print best hyperparameters
print("\n=== Best Hyperparameters ===")
best_params = optimizer.max
print(f"Best Average Validation Accuracy: {best_params['target']:.2f}%")
print(f"Best Parameters: {best_params['params']}")

# Train final model with best parameters
def train_final_model(params):
    print("\nTraining final model with best hyperparameters...")
    torch.manual_seed(42)

    lr = params['lr']
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])
    nHidden_index = int(params['nHidden_index'])
    nHidden = nHidden_candidates[nHidden_index]

    print(f"Selected nHidden configuration: {nHidden}")

    train_dataset = torch.utils.data.TensorDataset(train_images_augmented, train_labels_one_hot)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    final_model = BasicMLP(nHidden)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train(final_model, train_loader, criterion, optimizer, num_epochs)

    print("\nTesting the final model on the test set...")
    test_accuracy = test(final_model, test_loader)
    print(f'Final Test Accuracy with best parameters: {test_accuracy:.2f}%')

# Train the final model
train_final_model(best_params['params'])