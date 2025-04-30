import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import os # Add os import

# --- 导入 NumPy 版 function.py 组件 ---
try:
    from function import (
        load_mnist_data, BasicMLP_NumPy, # NumPy MLP
        mse_loss, mse_loss_derivative, # NumPy MSE Loss
        SGD_Optimizer, # NumPy Optimizer
        prepare_data_loaders, get_batch, one_hot_encode # NumPy data utils
    )
    print("成功从 function.py (NumPy 版) 导入模块。")
except ImportError as e:
    print(f"无法导入 function.py (NumPy 版): {e}")
    exit()
# --------------------------------------

# 移除 PyTorch Device 设置
device = 'cpu'
print("Using device: CPU (NumPy)")

# Load MNIST dataset (使用 function.py 的 load_mnist_data，注释掉旧的)
# def load_mnist_images(file_path):
#     ...
# def load_mnist_labels(file_path):
#     ...

# Data paths (保持不变)
train_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# --- 使用 NumPy prepare_data_loaders 加载和预处理数据 ---
print("使用 NumPy prepare_data_loaders 加载数据 (MSE -> one-hot)...")
config_load = {
    'model_type': 'mlp',
    'batch_size': 64, # 稍后可在优化中调整
    'loss_type': 'mse', # 指定 MSE 以便 prepare_data_loaders 进行 one-hot
    'data_dir': os.path.dirname(train_images_path) # 从路径获取目录
}
try:
    train_loader_info_full, test_loader_info = prepare_data_loaders(config_load)
    print(f"NumPy 数据加载完成。")
except FileNotFoundError as e:
     print(f"错误: {e}")
     exit()
# 移除旧的 PyTorch 加载和预处理代码
# train_images = load_mnist_images(train_images_path)
# ... (PyTorch tensor conversion and DataLoader)
# ---------------------------------------------------------

# Define MLP model with configurable nHidden (使用 NumPy BasicMLP_NumPy)
# 移除 PyTorch MLP 定义
# class BasicMLP(nn.Module):
#     ...
print("使用 NumPy 版 BasicMLP_NumPy 模型。")


# Training function (NumPy version for MSE - from v4)
def train_numpy(model, loader_info, loss_fn, loss_deriv_fn, optimizer, num_epochs):
    model.set_training_mode(True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        indices = np.arange(loader_info['num_samples'])
        if loader_info['shuffle']:
            np.random.shuffle(indices)
        for i in range(loader_info['num_batches']):
            batch_images, batch_labels_one_hot = get_batch(loader_info, i, indices)

            logits = model.forward(batch_images)
            loss = loss_fn(logits, batch_labels_one_hot)
            grad_loss = loss_deriv_fn(logits, batch_labels_one_hot)
            model.backward(grad_loss)
            optimizer.step()

            running_loss += loss
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{loader_info["num_batches"]}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    model.set_training_mode(False)

# Testing function (NumPy version - from v4)
def test_numpy(model, loader_info):
    model.set_training_mode(False)
    correct = 0
    total = 0
    for i in range(loader_info['num_batches']):
        images, labels_one_hot = get_batch(loader_info, i)
        logits = model.forward(images)
        predicted_indices = np.argmax(logits, axis=1)
        true_indices = np.argmax(labels_one_hot, axis=1) # Labels are one-hot for MSE
        total += true_indices.shape[0]
        correct += np.sum(predicted_indices == true_indices)
    accuracy = 100 * correct / total
    return accuracy

# Define nHidden candidates (保持不变)
nHidden_candidates = [
    [128],          # One hidden layer with 128 neurons
    [256, 128],     # Two hidden layers with 256 and 128 neurons
    [512, 256, 128] # Three hidden layers with 512, 256, and 128 neurons
]

# Objective function for Bayesian optimization (NumPy version, includes momentum)
def objective_numpy(lr, num_epochs, batch_size, k_folds, nHidden_index, momentum):
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    nHidden_index = int(np.round(nHidden_index))
    if not 0 <= nHidden_index < len(nHidden_candidates):
         nHidden_index = 0
    nHidden = nHidden_candidates[nHidden_index]
    # momentum = momentum # Already a float

    # Use full training data loaded earlier
    all_train_images = train_loader_info_full['images']
    all_train_labels = train_loader_info_full['labels'] # Already one-hot
    n_samples = train_loader_info_full['num_samples']
    batch_size_fold = int(batch_size)

    np.random.seed(42)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
        print(f'Fold {fold+1}/{k_folds}')
        # Create fold loader info (same as v4)
        fold_train_loader_info = {
            'images': all_train_images[train_idx],
            'labels': all_train_labels[train_idx],
            'batch_size': batch_size_fold,
            'num_samples': len(train_idx),
            'num_batches': int(np.ceil(len(train_idx) / batch_size_fold)),
            'shuffle': True,
            'use_one_hot': True,
            'num_classes': 10
        }
        fold_val_loader_info = {
            'images': all_train_images[val_idx],
            'labels': all_train_labels[val_idx],
            'batch_size': batch_size_fold,
            'num_samples': len(val_idx),
            'num_batches': int(np.ceil(len(val_idx) / batch_size_fold)),
            'shuffle': False,
            'use_one_hot': True,
            'num_classes': 10
        }

        model = BasicMLP_NumPy(nHidden=nHidden, activation_fn='sigmoid')
        loss_fn_mse, loss_deriv_fn_mse = mse_loss, mse_loss_derivative
        # Use momentum in NumPy optimizer
        optimizer = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

        train_numpy(model, fold_train_loader_info, loss_fn_mse, loss_deriv_fn_mse, optimizer, num_epochs)
        val_accuracy = test_numpy(model, fold_val_loader_info)
        fold_accuracies.append(val_accuracy)
        print(f'Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')

    avg_val_accuracy = np.mean(fold_accuracies)
    print(f'Average Validation Accuracy: {avg_val_accuracy:.2f}%')
    return avg_val_accuracy

# Set up Bayesian optimization (Add momentum bounds)
pbounds = {
    'lr': (0.001, 0.2),
    'num_epochs': (5, 15),
    'batch_size': (16, 128),
    'k_folds': (3, 5),
    'nHidden_index': (0, len(nHidden_candidates) - 0.01),
    'momentum': (0.0, 0.99)  # Momentum range (NumPy version handles 0)
}

bo_optimizer = BayesianOptimization(f=objective_numpy, pbounds=pbounds, random_state=42, verbose=2)

# Run Bayesian optimization
print("Starting Bayesian Optimization (NumPy with momentum)...")
bo_optimizer.maximize(init_points=5, n_iter=10)

# Print best hyperparameters
print("\n=== Best Hyperparameters (NumPy with momentum) ===")
best_params = bo_optimizer.max
print(f"Best Average Validation Accuracy: {best_params['target']:.2f}%")
print(f"Best Parameters: {best_params['params']}")

# Train final model with best parameters (NumPy version, includes momentum)
def train_final_model_numpy(params):
    print("\nTraining final model with best hyperparameters (NumPy with momentum)...")
    np.random.seed(42)

    lr = params['lr']
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])
    nHidden_index = int(np.round(params['nHidden_index']))
    if not 0 <= nHidden_index < len(nHidden_candidates):
         nHidden_index = 0
    nHidden = nHidden_candidates[nHidden_index]
    momentum = params['momentum']

    print(f"Selected nHidden configuration: {nHidden}")
    print(f"Selected momentum: {momentum}")

    # Create final train loader info
    final_train_loader_info = train_loader_info_full.copy()
    final_train_loader_info['batch_size'] = batch_size
    final_train_loader_info['num_batches'] = int(np.ceil(final_train_loader_info['num_samples'] / batch_size))

    final_model = BasicMLP_NumPy(nHidden=nHidden, activation_fn='sigmoid')
    loss_fn_mse, loss_deriv_fn_mse = mse_loss, mse_loss_derivative
    # Pass momentum to final optimizer
    optimizer = SGD_Optimizer(final_model, learning_rate=lr, momentum=momentum)

    train_numpy(final_model, final_train_loader_info, loss_fn_mse, loss_deriv_fn_mse, optimizer, num_epochs)

    print("\nTesting the final model on the test set (NumPy)...")
    test_accuracy = test_numpy(final_model, test_loader_info)
    print(f'Final Test Accuracy with best parameters: {test_accuracy:.2f}%')

# Train the final model with best parameters
train_final_model_numpy(best_params['params'])