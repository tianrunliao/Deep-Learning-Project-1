import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os # Add os import
import sys # Import sys for exit

# --- 导入 NumPy 版 function.py 组件 ---
try:
    # 使用 function.py 中正确的类名
    from function import (
        prepare_data_loaders, get_batch, # NumPy data utils
        cross_entropy_loss, cross_entropy_loss_derivative, # NumPy Cross Entropy Loss
        SGD_Optimizer, # NumPy Optimizer
        ConvLayer, PoolingLayer, ActivationLayer, FlattenLayer, LinearLayer, DropoutLayer, # NumPy Layers (Correct Names, ActivationLayer instead of ReLULayer)
        BasicCNN_NumPy # Import the NumPy CNN definition from function.py
    )
    print("成功从 function.py (NumPy 版) 导入模块。")
except ImportError as e:
    print(f"无法从 function.py (NumPy 版) 导入必要的模块: {e}")
    print("请确保 function.py 文件存在，并且包含以下 NumPy 类：")
    print("ConvLayer, PoolingLayer, ActivationLayer, FlattenLayer, LinearLayer, DropoutLayer, BasicCNN_NumPy, SGD_Optimizer, etc.")
    sys.exit(1) # 导入失败则退出
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
    'model_type': 'cnn',
    'batch_size': 64,
    'loss_type': 'cross_entropy',
    'data_dir': os.path.dirname(train_images_path)
}
try:
    train_loader_info_full, test_loader_info = prepare_data_loaders(config_load)
    print(f"NumPy 数据加载完成。训练数据形状: {train_loader_info_full['images'].shape}, 测试数据形状: {test_loader_info['images'].shape}")
except FileNotFoundError as e:
     print(f"错误: {e}")
     exit()
# 移除旧的 PyTorch 加载和预处理代码
# ...
# ---------------------------------------------------------------------------

# 移除 PyTorch CNN 定义
# class BasicCNN(nn.Module): ...
# BasicCNN_NumPy 应该已从 function.py 成功导入

print("使用导入的 NumPy 版 BasicCNN_NumPy 模型。")

# Training function (NumPy version for Cross Entropy - adapted for CNN from v8)
def train_numpy_ce(model, loader_info, val_loader_info, loss_fn, loss_deriv_fn, optimizer, num_epochs, regularization=None, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.set_training_mode(True)
        running_loss = 0.0
        indices = np.arange(loader_info['num_samples'])
        if loader_info['shuffle']:
            np.random.shuffle(indices)
        # Use tqdm for batch iteration
        for i in tqdm(range(loader_info['num_batches']), desc=f'Epoch {epoch+1}', leave=False):
            batch_images, batch_labels_indices = get_batch(loader_info, i, indices)
            logits = model.forward(batch_images)
            loss = loss_fn(logits, batch_labels_indices)
            grad_loss = loss_deriv_fn(logits, batch_labels_indices)
            model.backward(grad_loss)
            optimizer.step()
            running_loss += loss
            # Optional: Log less frequently inside tqdm loop
            # if (i + 1) % 100 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{loader_info["num_batches"]}], Avg Loss: {running_loss/(i+1):.4f}')
        epoch_avg_loss = running_loss / loader_info['num_batches']
        print(f'Epoch [{epoch+1}/{num_epochs}] Completed, Average Loss: {epoch_avg_loss:.4f}')

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

# Testing function (NumPy version for Cross Entropy - adapted for CNN from v8)
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

# 目标函数（贝叶斯优化 - NumPy CNN version from v8)
def objective_numpy_cnn(lr, num_epochs, batch_size, k_folds, conv_filters_index, momentum, dropout_rate):
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    conv_filters_index = int(np.round(conv_filters_index))
    if not 0 <= conv_filters_index < len(conv_filters_candidates):
        conv_filters_index = 0
    conv_filters = conv_filters_candidates[conv_filters_index]

    regularization_types = ['none', 'dropout', 'early_stopping']
    fold_accuracies = {reg: [] for reg in regularization_types}

    all_train_images = train_loader_info_full['images']
    all_train_labels = train_loader_info_full['labels']
    n_samples = train_loader_info_full['num_samples']
    batch_size_fold = int(batch_size)

    np.random.seed(42)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Use tqdm for KFold loop
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(np.arange(n_samples)), total=k_folds, desc='Folds')):
        # print(f'Fold {fold+1}/{k_folds}') # tqdm provides progress
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
            print(f'Fold {fold+1}, Training with {reg} regularization')
            current_dropout_rate = dropout_rate if reg == 'dropout' else 0.0
            model = BasicCNN_NumPy(conv_filters, dropout_rate=current_dropout_rate)

            loss_fn_ce, loss_deriv_fn_ce = cross_entropy_loss, cross_entropy_loss_derivative
            optimizer = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

            train_numpy_ce(model, fold_train_loader_info, fold_val_loader_info if reg == 'early_stopping' else None,
                           loss_fn_ce, loss_deriv_fn_ce, optimizer, num_epochs, regularization=reg, patience=5)
            val_accuracy = test_numpy_ce(model, fold_val_loader_info)
            fold_accuracies[reg].append(val_accuracy)
            print(f'Fold {fold+1} {reg} Validation Accuracy: {val_accuracy:.2f}%')

    avg_val_accuracies = {reg: np.mean(acc) for reg, acc in fold_accuracies.items()}
    for reg, avg_acc in avg_val_accuracies.items():
        print(f'Average {reg} Validation Accuracy: {avg_acc:.2f}%')
    return avg_val_accuracies.get('none', 0.0)

# 定义贝叶斯优化的超参数边界 (NumPy CNN version from v8)
pbounds = {
    'lr': (0.001, 0.1),
    'num_epochs': (5, 15),
    'batch_size': (16, 128),
    'k_folds': (3, 5),
    'conv_filters_index': (0, len(conv_filters_candidates) - 0.01),
    'momentum': (0.0, 0.99),
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

# 可视化卷积核的函数 (NumPy version)
def visualize_conv_kernels(model):
    # Find the first ConvLayer instance (使用正确的类名 ConvLayer)
    first_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, ConvLayer): # 使用正确的类名
            first_conv_layer = layer
            break
    
    if first_conv_layer is None or not hasattr(first_conv_layer, 'weights'):
        print("无法找到第一个卷积层或其权重。")
        return
        
    # Assuming weights are stored in layer.weights with shape (out_channels, in_channels, K, K)
    conv1_weights = first_conv_layer.weights
    
    if conv1_weights is None or conv1_weights.ndim != 4:
         print(f"第一个卷积层的权重形状不符合预期 (预期 4D, 得到 {conv1_weights.ndim}D)。")
         return
         
    num_filters = conv1_weights.shape[0]
    print(f"可视化第一个卷积层的 {num_filters} 个滤波器...")
    
    # 设置画布大小
    cols = 8 # Adjust layout as needed
    rows = int(np.ceil(num_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten() # Flatten to easily iterate

    for i in range(num_filters):
        if i < len(axes):
            # Assuming input channel is 1 (grayscale)
            kernel = conv1_weights[i, 0, :, :]
            axes[i].imshow(kernel, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'F {i+1}')
            
    # Hide unused subplots
    for j in range(num_filters, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# 使用最佳超参数训练最终模型并比较不同正则化方式，同时可视化卷积核 (NumPy CNN version)
def train_final_model_numpy_cnn(params):
    print("\nTraining final model with best hyperparameters (NumPy CNN)...")
    np.random.seed(42)

    lr = params['lr']
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])
    conv_filters_index = int(np.round(params['conv_filters_index']))
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
    final_model = None # Store the model trained without regularization for visualization

    for reg in regularization_types:
        print(f"\nRunning final model with {reg} regularization")
        current_dropout_rate = dropout_rate if reg == 'dropout' else 0.0
        model = BasicCNN_NumPy(conv_filters, dropout_rate=current_dropout_rate)

        loss_fn_ce, loss_deriv_fn_ce = cross_entropy_loss, cross_entropy_loss_derivative
        optimizer = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

        train_numpy_ce(model, final_train_loader_info, final_val_loader_info if reg == 'early_stopping' else None,
                       loss_fn_ce, loss_deriv_fn_ce, optimizer, num_epochs, regularization=reg, patience=5)
        test_accuracy = test_numpy_ce(model, final_test_loader_info)
        results[reg] = test_accuracy
        print(f'Final Test Accuracy with {reg}: {test_accuracy:.2f}%')
        
        # Save the model trained without regularization for visualization
        if reg == 'none':
            final_model = model 

    # Create and print results table
    df = pd.DataFrame(list(results.items()), columns=['Regularization', 'Test Accuracy (%)'])
    print("\n=== Final Results Table (NumPy CNN) ===")
    print(df)
    
    # Visualize kernels of the final model (trained without regularization)
    if final_model:
        print("\nVisualizing kernels of the final model (no regularization)...")
        visualize_conv_kernels(final_model)
    else:
        print("\n无法可视化卷积核，因为没有训练好的'none'正则化模型。")

# Train the final model with best parameters using the NumPy CNN version
train_final_model_numpy_cnn(best_params['params'])