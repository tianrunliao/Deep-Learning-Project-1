import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import pandas as pd
import os

# --- 导入 NumPy 版 function.py 组件 ---
try:
    from function import (
        load_mnist_data, BasicMLP_NumPy, # NumPy MLP
        mse_loss, mse_loss_derivative, # NumPy MSE Loss
        SGD_Optimizer, # NumPy Optimizer
        prepare_data_loaders, get_batch, one_hot_encode, # NumPy data utils
        LinearLayer # Import LinearLayer to access weights for L1
    )
    print("成功从 function.py (NumPy 版) 导入模块。")
except ImportError as e:
    print(f"无法导入 function.py (NumPy 版): {e}")
    exit()
# --------------------------------------

# 移除 PyTorch Device 设置
device = 'cpu'
print("Using device: CPU (NumPy)")

# Load MNIST dataset (使用 function.py 的 load_mnist_data)
# ... (注释掉旧的加载函数)

# Data paths (保持不变)
train_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# --- 使用 NumPy prepare_data_loaders 加载和预处理数据 ---
print("使用 NumPy prepare_data_loaders 加载数据 (MSE -> one-hot)...")
config_load = {
    'model_type': 'mlp',
    'batch_size': 64,
    'loss_type': 'mse',
    'data_dir': os.path.dirname(train_images_path)
}
try:
    train_loader_info_full, test_loader_info = prepare_data_loaders(config_load)
    print(f"NumPy 数据加载完成。")
except FileNotFoundError as e:
     print(f"错误: {e}")
     exit()
# 移除旧的 PyTorch 加载代码
# ...
# ---------------------------------------------------------

# Define MLP model with configurable nHidden and Dropout (使用 NumPy BasicMLP_NumPy)
# 移除 PyTorch MLP 定义
# class BasicMLP(nn.Module):
#     ...
print("使用 NumPy 版 BasicMLP_NumPy 模型。")


# Training function with regularization and early stopping (NumPy version for MSE - from v6)
def train_numpy(model, loader_info, val_loader_info, loss_fn, loss_deriv_fn, optimizer, num_epochs, regularization=None, l1_lambda=0.0, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.set_training_mode(True)
        running_loss = 0.0
        indices = np.arange(loader_info['num_samples'])
        if loader_info['shuffle']:
            np.random.shuffle(indices)

        for i in range(loader_info['num_batches']):
            batch_images, batch_labels_one_hot = get_batch(loader_info, i, indices)

            logits = model.forward(batch_images)
            loss = loss_fn(logits, batch_labels_one_hot)

            if regularization == 'l1' and l1_lambda > 0:
                l1_norm = 0
                for layer in model.layers:
                    if hasattr(layer, 'weights') and layer.weights is not None:
                        l1_norm += np.sum(np.abs(layer.weights))
                loss += l1_lambda * l1_norm

            grad_loss = loss_deriv_fn(logits, batch_labels_one_hot)
            model.backward(grad_loss)
            optimizer.step()

            running_loss += loss
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{loader_info["num_batches"]}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        if regularization == 'early_stopping' and val_loader_info:
            model.set_training_mode(False)
            val_loss = 0.0
            for i in range(val_loader_info['num_batches']):
                val_images, val_labels = get_batch(val_loader_info, i)
                val_logits = model.forward(val_images)
                batch_val_loss = loss_fn(val_logits, val_labels)
                val_loss += batch_val_loss
            epoch_val_loss = val_loss / val_loader_info['num_batches']
            print(f'Validation Loss: {epoch_val_loss:.4f}')
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
    model.set_training_mode(False)

# Testing function (NumPy version - from v6)
def test_numpy(model, loader_info):
    model.set_training_mode(False)
    correct = 0
    total = 0
    for i in range(loader_info['num_batches']):
        images, labels_one_hot = get_batch(loader_info, i)
        logits = model.forward(images)
        predicted_indices = np.argmax(logits, axis=1)
        true_indices = np.argmax(labels_one_hot, axis=1)
        total += true_indices.shape[0]
        correct += np.sum(predicted_indices == true_indices)
    accuracy = 100 * correct / total
    return accuracy

# Define nHidden candidates (保持不变)
nHidden_candidates = [
    [128],
    [256, 128],
    [512, 256, 128]
]

# Objective function for Bayesian optimization with regularization (NumPy version)
# Removed weight_decay from arguments and logic
def objective_numpy(lr, num_epochs, batch_size, k_folds, nHidden_index, momentum, dropout_rate, l1_lambda):
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    nHidden_index = int(np.round(nHidden_index))
    if not 0 <= nHidden_index < len(nHidden_candidates):
        nHidden_index = 0
    nHidden = nHidden_candidates[nHidden_index]

    # L2 (weight_decay) is removed
    regularization_types = ['none', 'l1', 'dropout', 'early_stopping']
    fold_accuracies = {reg: [] for reg in regularization_types}

    all_train_images = train_loader_info_full['images']
    all_train_labels = train_loader_info_full['labels']
    n_samples = train_loader_info_full['num_samples']
    batch_size_fold = int(batch_size)

    np.random.seed(42)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
        print(f'Fold {fold+1}/{k_folds}')
        # Create fold loader info
        fold_train_loader_info = {
            'images': all_train_images[train_idx], 'labels': all_train_labels[train_idx],
            'batch_size': batch_size_fold, 'num_samples': len(train_idx),
            'num_batches': int(np.ceil(len(train_idx) / batch_size_fold)),
            'shuffle': True, 'use_one_hot': True, 'num_classes': 10
        }
        fold_val_loader_info = {
            'images': all_train_images[val_idx], 'labels': all_train_labels[val_idx],
            'batch_size': batch_size_fold, 'num_samples': len(val_idx),
            'num_batches': int(np.ceil(len(val_idx) / batch_size_fold)),
            'shuffle': False, 'use_one_hot': True, 'num_classes': 10
        }

        for reg in regularization_types:
            print(f'Training with {reg} regularization')
            current_dropout_rate = dropout_rate if reg == 'dropout' else 0.0
            model = BasicMLP_NumPy(nHidden=nHidden, dropout_rate=current_dropout_rate, activation_fn='sigmoid')

            loss_fn_mse, loss_deriv_fn_mse = mse_loss, mse_loss_derivative
            # Optimizer doesn't support weight_decay
            optimizer = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

            # Train
            train_numpy(model, fold_train_loader_info, fold_val_loader_info if reg == 'early_stopping' else None,
                  loss_fn_mse, loss_deriv_fn_mse, optimizer, num_epochs, regularization=reg, l1_lambda=l1_lambda if reg == 'l1' else 0.0, patience=5)
            val_accuracy = test_numpy(model, fold_val_loader_info)
            fold_accuracies[reg].append(val_accuracy)
            print(f'Fold {fold+1} {reg} Validation Accuracy: {val_accuracy:.2f}%')

    avg_val_accuracies = {reg: np.mean(acc) for reg, acc in fold_accuracies.items()}
    for reg, avg_acc in avg_val_accuracies.items():
        print(f'Average {reg} Validation Accuracy: {avg_acc:.2f}%')
    # Optimize based on no regularization as baseline
    return avg_val_accuracies.get('none', 0.0)

# Set up Bayesian optimization with extended hyperparameter space (remove weight_decay)
pbounds = {
    'lr': (0.001, 0.2),
    'num_epochs': (5, 15),
    'batch_size': (16, 128),
    'k_folds': (3, 5),
    'nHidden_index': (0, len(nHidden_candidates) - 0.01),
    'momentum': (0.0, 0.99),
    'dropout_rate': (0.0, 0.5),
    'l1_lambda': (0.00001, 0.001)
    # 'weight_decay': (0.0001, 0.01) # Removed
}

bo_optimizer = BayesianOptimization(f=objective_numpy, pbounds=pbounds, random_state=42, verbose=2)

# Run Bayesian optimization
print("Starting Bayesian Optimization (NumPy - Adjusted)...")
bo_optimizer.maximize(init_points=5, n_iter=10)

# Print best hyperparameters
print("\n=== Best Hyperparameters (NumPy - Adjusted) ===")
best_params = bo_optimizer.max
print(f"Best Average Validation Accuracy (no regularization): {best_params['target']:.2f}%")
print(f"Best Parameters: {best_params['params']}")

# Train final model with best parameters and compare regularizations (NumPy version)
def train_final_model_numpy(params):
    print("\nTraining final model with best hyperparameters (NumPy - Adjusted)...")
    np.random.seed(42)

    lr = params['lr']
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])
    nHidden_index = int(np.round(params['nHidden_index']))
    if not 0 <= nHidden_index < len(nHidden_candidates):
        nHidden_index = 0
    nHidden = nHidden_candidates[nHidden_index]
    momentum = params['momentum']
    dropout_rate = params['dropout_rate']
    l1_lambda = params['l1_lambda']
    # weight_decay removed

    print(f"Selected nHidden configuration: {nHidden}")
    print(f"Selected momentum: {momentum}")
    print(f"Selected dropout_rate: {dropout_rate}")
    print(f"Selected l1_lambda: {l1_lambda}")
    # print(f"Selected weight_decay: {weight_decay}") # Removed

    # Final loader info
    final_train_loader_info = train_loader_info_full.copy()
    final_train_loader_info['batch_size'] = batch_size
    final_train_loader_info['num_batches'] = int(np.ceil(final_train_loader_info['num_samples'] / batch_size))
    final_val_loader_info = final_train_loader_info # Use train as val

    regularization_types = ['none', 'l1', 'dropout', 'early_stopping'] # Removed 'l2'
    results = {}

    for reg in regularization_types:
        print(f"\nRunning final model with {reg} regularization")
        current_dropout_rate = dropout_rate if reg == 'dropout' else 0.0
        model = BasicMLP_NumPy(nHidden=nHidden, dropout_rate=current_dropout_rate, activation_fn='sigmoid')

        loss_fn_mse, loss_deriv_fn_mse = mse_loss, mse_loss_derivative
        # Optimizer doesn't support weight_decay
        optimizer = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

        # Train
        train_numpy(model, final_train_loader_info, final_val_loader_info if reg == 'early_stopping' else None,
              loss_fn_mse, loss_deriv_fn_mse, optimizer, num_epochs, regularization=reg, l1_lambda=l1_lambda if reg == 'l1' else 0.0, patience=5)

        # Test
        test_accuracy = test_numpy(model, test_loader_info)
        results[reg] = test_accuracy
        print(f'Final Test Accuracy with {reg}: {test_accuracy:.2f}%')

    # Create and print results table
    df = pd.DataFrame(list(results.items()), columns=['Regularization', 'Test Accuracy (%)'])
    print("\n=== Final Results Table (NumPy - L2 Excluded) ===")
    print(df)

# Train the final model with best parameters
train_final_model_numpy(best_params['params'])