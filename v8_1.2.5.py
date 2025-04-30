import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import pandas as pd
import os # Add os import

# --- 导入 NumPy 版 function.py 组件 ---
try:
    from function import (
        prepare_data_loaders, get_batch, # NumPy data utils
        cross_entropy_loss, cross_entropy_loss_derivative, # NumPy Cross Entropy Loss
        SGD_Optimizer, # NumPy Optimizer
        ConvLayer_NumPy, MaxPoolLayer_NumPy, ReLULayer, FlattenLayer_NumPy, LinearLayer, # NumPy Layers
        BasicMLP_NumPy # Although defining CNN, keep MLP import for potential reference/consistency
    )
    print("成功从 function.py (NumPy 版) 导入模块。")
except ImportError as e:
    print(f"无法导入 function.py (NumPy 版): {e}")
    exit()
# --------------------------------------

# 移除 PyTorch Device 设置 / GPU 支持
device = 'cpu'
print("Using device: CPU (NumPy)")

# 移除 PyTorch MNIST 加载函数
# def load_mnist_images(file_path): ...
# def load_mnist_labels(file_path): ...

# 数据路径 (保持不变)
train_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# --- 使用 NumPy prepare_data_loaders 加载和预处理数据 (CNN -> N, C, H, W) ---
print("使用 NumPy prepare_data_loaders 加载数据 (CNN, Cross Entropy)...")
config_load = {
    'model_type': 'cnn', # Specify model type for potential shape adjustments
    'batch_size': 64,
    'loss_type': 'cross_entropy',
    'data_dir': os.path.dirname(train_images_path)
}
try:
    train_loader_info_full, test_loader_info = prepare_data_loaders(config_load)
    print(f"NumPy 数据加载完成。训练数据形状: {train_loader_info_full['images'].shape}, 测试数据形状: {test_loader_info['images'].shape}")
    print(f"训练标签形状: {train_loader_info_full['labels'].shape}") # Check labels are indices
except FileNotFoundError as e:
     print(f"错误: {e}")
     exit()
# 移除旧的 PyTorch 加载和预处理代码
# ...
# ---------------------------------------------------------------------------

# 移除 PyTorch CNN 定义
# class BasicCNN(nn.Module): ...

# 定义 NumPy 版 CNN 模型
class BasicCNN_NumPy:
    def __init__(self, conv_filters, dropout_rate=0.0):
        # Expected input shape: (N, C, H, W) = (N, 1, 28, 28)
        # conv_filters: e.g., [16, 32]
        self.layers = [
            # Layer 1: Conv -> ReLU -> Pool
            ConvLayer_NumPy(input_channels=1, output_channels=conv_filters[0], kernel_size=3, padding=1),
            ReLULayer(),
            MaxPoolLayer_NumPy(kernel_size=2, stride=2), # Output: (N, filters[0], 14, 14)
            
            # Layer 2: Conv -> ReLU -> Pool
            ConvLayer_NumPy(input_channels=conv_filters[0], output_channels=conv_filters[1], kernel_size=3, padding=1),
            ReLULayer(),
            MaxPoolLayer_NumPy(kernel_size=2, stride=2), # Output: (N, filters[1], 7, 7)
            
            # Flatten
            FlattenLayer_NumPy(), # Output: (N, filters[1] * 7 * 7)
            
            # FC Layers: Linear -> ReLU -> Dropout -> Linear -> Softmax (for cross-entropy)
            LinearLayer(input_size=conv_filters[1] * 7 * 7, output_size=128),
            ReLULayer(),
            # Note: DropoutLayer_NumPy needs implementation or adaptation in function.py
            # Assuming dropout is handled within LinearLayer or a separate DropoutLayer exists
            # For now, conditionally add LinearLayer for dropout effect simulation if rate > 0
            # Proper DropoutLayer needed for real dropout
            # If function.py has DropoutLayer_NumPy, use it here.
            # Temporary: Skip dropout if not readily available/verified in function.py
            LinearLayer(input_size=128, output_size=10), # Output: (N, 10) - Raw logits expected by CE loss
            # Softmax is typically combined with cross-entropy loss function, so output raw logits
        ]
        # Add dropout layer properly if available and verified
        # Example: if dropout_rate > 0 and hasattr(function, 'DropoutLayer_NumPy'):
        #    self.layers.insert(-1, function.DropoutLayer_NumPy(dropout_rate))
        
        self.params = []
        self.grads = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                self.params.extend(layer.params)
                self.grads.extend(layer.grads)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def set_training_mode(self, training):
        # Implement if dropout or batchnorm layers need mode switching
        pass

print("使用 NumPy 版 BasicCNN_NumPy 模型。")

# Training function (NumPy version for Cross Entropy - adapted for CNN)
# Reusing train_numpy_ce name, but ensure data shapes are correct
def train_numpy_ce(model, loader_info, val_loader_info, loss_fn, loss_deriv_fn, optimizer, num_epochs, regularization=None, l1_lambda=0.0, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.set_training_mode(True)
        running_loss = 0.0
        indices = np.arange(loader_info['num_samples'])
        if loader_info['shuffle']:
            np.random.shuffle(indices)
        for i in range(loader_info['num_batches']):
            batch_images, batch_labels_indices = get_batch(loader_info, i, indices)
            # Ensure batch_images has shape (N, C, H, W) if needed by ConvLayer_NumPy
            # prepare_data_loaders should handle this if model_type='cnn'
            logits = model.forward(batch_images)
            loss = loss_fn(logits, batch_labels_indices)
            # L1 regularization is removed as per original v8 logic
            grad_loss = loss_deriv_fn(logits, batch_labels_indices)
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
                val_images, val_labels_indices = get_batch(val_loader_info, i)
                val_logits = model.forward(val_images)
                batch_val_loss = loss_fn(val_logits, val_labels_indices)
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

# Testing function (NumPy version for Cross Entropy - adapted for CNN)
# Reusing test_numpy_ce name
def test_numpy_ce(model, loader_info):
    model.set_training_mode(False)
    correct = 0
    total = 0
    for i in range(loader_info['num_batches']):
        images, labels_indices = get_batch(loader_info, i)
        logits = model.forward(images)
        predicted_indices = np.argmax(logits, axis=1)
        total += labels_indices.shape[0]
        correct += np.sum(predicted_indices == labels_indices)
    accuracy = 100 * correct / total
    return accuracy

# 定义不同规模卷积层配置 (保持不变)
conv_filters_candidates = [
    [16, 32],
    [32, 64],
    [64, 128]
]

# 目标函数（贝叶斯优化 - NumPy CNN version）
def objective_numpy_cnn(lr, num_epochs, batch_size, k_folds, conv_filters_index, momentum, dropout_rate):
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    conv_filters_index = int(np.round(conv_filters_index)) # Use round
    if not 0 <= conv_filters_index < len(conv_filters_candidates):
        conv_filters_index = 0
    conv_filters = conv_filters_candidates[conv_filters_index]

    # L1/L2 removed in v8 original logic, only dropout/early stopping considered
    regularization_types = ['none', 'dropout', 'early_stopping']
    fold_accuracies = {reg: [] for reg in regularization_types}

    all_train_images = train_loader_info_full['images']
    all_train_labels = train_loader_info_full['labels']
    n_samples = train_loader_info_full['num_samples']
    batch_size_fold = int(batch_size)

    np.random.seed(42)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
        print(f'Fold {fold+1}/{k_folds}')
        fold_train_loader_info = {
            'images': all_train_images[train_idx], 'labels': all_train_labels[train_idx],
            'batch_size': batch_size_fold, 'num_samples': len(train_idx),
            'num_batches': int(np.ceil(len(train_idx) / batch_size_fold)),
            'shuffle': True, 'use_one_hot': False, 'model_type': 'cnn'
        }
        fold_val_loader_info = {
            'images': all_train_images[val_idx], 'labels': all_train_labels[val_idx],
            'batch_size': batch_size_fold, 'num_samples': len(val_idx),
            'num_batches': int(np.ceil(len(val_idx) / batch_size_fold)),
            'shuffle': False, 'use_one_hot': False, 'model_type': 'cnn'
        }

        for reg in regularization_types:
            print(f'Training with {reg} regularization')
            # Use BasicCNN_NumPy, pass dropout_rate (implementation detail handled in class)
            current_dropout_rate = dropout_rate if reg == 'dropout' else 0.0
            # Pass dropout rate, but actual implementation in BasicCNN_NumPy needs verification
            model = BasicCNN_NumPy(conv_filters, dropout_rate=current_dropout_rate)

            loss_fn_ce, loss_deriv_fn_ce = cross_entropy_loss, cross_entropy_loss_derivative
            optimizer = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

            # Use train_numpy_ce and test_numpy_ce (already adapted for CE)
            train_numpy_ce(model, fold_train_loader_info, fold_val_loader_info if reg == 'early_stopping' else None,
                           loss_fn_ce, loss_deriv_fn_ce, optimizer, num_epochs, regularization=reg, l1_lambda=0.0, patience=5)
            val_accuracy = test_numpy_ce(model, fold_val_loader_info)
            fold_accuracies[reg].append(val_accuracy)
            print(f'Fold {fold+1} {reg} Validation Accuracy: {val_accuracy:.2f}%')

    avg_val_accuracies = {reg: np.mean(acc) for reg, acc in fold_accuracies.items()}
    for reg, avg_acc in avg_val_accuracies.items():
        print(f'Average {reg} Validation Accuracy: {avg_acc:.2f}%')
    return avg_val_accuracies.get('none', 0.0)

# 定义贝叶斯优化的超参数边界 (NumPy CNN version)
pbounds = {
    'lr': (0.001, 0.1),
    'num_epochs': (5, 15),
    'batch_size': (16, 128),
    'k_folds': (3, 5),
    'conv_filters_index': (0, len(conv_filters_candidates) - 0.01), # Use float index
    'momentum': (0.0, 0.99), # Adjusted range
    'dropout_rate': (0.0, 0.5)
}

# Use the NumPy CNN objective function
bo_optimizer = BayesianOptimization(f=objective_numpy_cnn, pbounds=pbounds, random_state=42, verbose=2)

# 执行贝叶斯优化
print("Starting Bayesian Optimization (NumPy CNN)...")
bo_optimizer.maximize(init_points=5, n_iter=10)

# 打印最佳超参数
print("\n=== Best Hyperparameters (NumPy CNN) ===")
best_params = bo_optimizer.max
print(f"Best Average Validation Accuracy (no regularization): {best_params['target']:.2f}%")
print(f"Best Parameters: {best_params['params']}")

# 使用最佳超参数训练最终模型并比较不同正则化方式 (NumPy CNN version)
def train_final_model_numpy_cnn(params):
    print("\nTraining final model with best hyperparameters (NumPy CNN)...")
    np.random.seed(42)

    lr = params['lr']
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])
    conv_filters_index = int(np.round(params['conv_filters_index'])) # Use round
    if not 0 <= conv_filters_index < len(conv_filters_candidates):
        conv_filters_index = 0
    momentum = params['momentum']
    dropout_rate = params['dropout_rate']
    conv_filters = conv_filters_candidates[conv_filters_index]

    print(f"Selected conv_filters configuration: {conv_filters}")
    print(f"Selected momentum: {momentum}")
    print(f"Selected dropout_rate: {dropout_rate}")

    # Final loader info (CNN)
    final_train_loader_info = train_loader_info_full.copy()
    final_train_loader_info['batch_size'] = batch_size
    final_train_loader_info['num_batches'] = int(np.ceil(final_train_loader_info['num_samples'] / batch_size))
    final_train_loader_info['shuffle'] = True
    final_train_loader_info['use_one_hot'] = False
    final_train_loader_info['model_type'] = 'cnn'
    final_val_loader_info = final_train_loader_info

    final_test_loader_info = test_loader_info.copy()
    final_test_loader_info['batch_size'] = batch_size
    final_test_loader_info['num_batches'] = int(np.ceil(final_test_loader_info['num_samples'] / batch_size))
    final_test_loader_info['use_one_hot'] = False
    final_test_loader_info['model_type'] = 'cnn'

    regularization_types = ['none', 'dropout', 'early_stopping']
    results = {}

    for reg in regularization_types:
        print(f"\nRunning final model with {reg} regularization")
        current_dropout_rate = dropout_rate if reg == 'dropout' else 0.0
        model = BasicCNN_NumPy(conv_filters, dropout_rate=current_dropout_rate)

        loss_fn_ce, loss_deriv_fn_ce = cross_entropy_loss, cross_entropy_loss_derivative
        optimizer = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

        train_numpy_ce(model, final_train_loader_info, final_val_loader_info if reg == 'early_stopping' else None,
                       loss_fn_ce, loss_deriv_fn_ce, optimizer, num_epochs, regularization=reg, l1_lambda=0.0, patience=5)
        test_accuracy = test_numpy_ce(model, final_test_loader_info)
        results[reg] = test_accuracy
        print(f'Final Test Accuracy with {reg}: {test_accuracy:.2f}%')

    # Create and print results table
    df = pd.DataFrame(list(results.items()), columns=['Regularization', 'Test Accuracy (%)'])
    print("\n=== Final Results Table (NumPy CNN) ===")
    print(df)

# Train the final model with best parameters using the NumPy CNN version
train_final_model_numpy_cnn(best_params['params'])