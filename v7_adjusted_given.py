import numpy as np
import gzip
import struct
import pandas as pd
import os

# --- 导入 NumPy 版 function.py 组件 ---
try:
    from function import (
        load_mnist_data, BasicMLP_NumPy, # NumPy MLP
        cross_entropy_loss, cross_entropy_loss_derivative, # NumPy Cross Entropy Loss
        SGD_Optimizer, # NumPy Optimizer
        prepare_data_loaders, get_batch, # NumPy data utils (no one_hot_encode needed)
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

# 移除 PyTorch MNIST 加载函数
# def load_mnist_images(file_path): ...
# def load_mnist_labels(file_path): ...

# Data paths (保持不变)
train_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# --- 使用 NumPy prepare_data_loaders 加载和预处理数据 (Cross Entropy -> label indices) ---
print("使用 NumPy prepare_data_loaders 加载数据 (Cross Entropy -> label indices)...")
config_load = {
    'model_type': 'mlp',
    'batch_size': 64, # Default batch size, will be overridden later
    'loss_type': 'cross_entropy',
    'data_dir': os.path.dirname(train_images_path)
}
try:
    train_loader_info_full, test_loader_info = prepare_data_loaders(config_load)
    print(f"NumPy 数据加载完成。训练标签类型: {train_loader_info_full['labels'].dtype}, 测试标签类型: {test_loader_info['labels'].dtype}")
except FileNotFoundError as e:
     print(f"错误: {e}")
     exit()
# 移除旧的 PyTorch 加载和预处理代码
# ...
# ------------------------------------------------------------------------------------

# 移除 PyTorch MLP 定义
# class BasicMLP(nn.Module): ...
print("使用 NumPy 版 BasicMLP_NumPy 模型。")

# Training function (NumPy version for Cross Entropy - adapted from v7_1.2.4)
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
            logits = model.forward(batch_images)
            loss = loss_fn(logits, batch_labels_indices)
            if regularization == 'l1' and l1_lambda > 0:
                l1_norm = 0
                for layer in model.layers:
                    if hasattr(layer, 'weights') and layer.weights is not None:
                        l1_norm += np.sum(np.abs(layer.weights))
                loss += l1_lambda * l1_norm
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

# Testing function (NumPy version for Cross Entropy - adapted from v7_1.2.4)
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

# Define nHidden candidates (保持不变)
nHidden_candidates = [
    [128],
    [256, 128],
    [512, 256, 128]
]

# Train final model with fixed hyperparameters and compare regularizations (NumPy version)
def train_final_model_numpy_ce():
    print("\nTraining final model with fixed hyperparameters (NumPy - Cross Entropy)...")
    np.random.seed(42)

    # Fixed hyperparameters (L2 weight_decay removed)
    lr = 0.01
    num_epochs = 10
    batch_size = 64
    nHidden_index = 1  # Choose [256, 128]
    momentum = 0.9
    dropout_rate = 0.3
    l1_lambda = 0.0001
    # weight_decay = 0.0001 # Removed
    nHidden = nHidden_candidates[nHidden_index]

    print(f"Selected nHidden configuration: {nHidden}")
    print(f"Selected momentum: {momentum}")
    print(f"Selected dropout_rate: {dropout_rate}")
    print(f"Selected l1_lambda: {l1_lambda}")
    # print(f"Selected weight_decay: {weight_decay}") # Removed

    # Final loader info (using label indices and fixed batch size)
    final_train_loader_info = train_loader_info_full.copy()
    final_train_loader_info['batch_size'] = batch_size
    final_train_loader_info['num_batches'] = int(np.ceil(final_train_loader_info['num_samples'] / batch_size))
    final_train_loader_info['shuffle'] = True
    final_train_loader_info['use_one_hot'] = False
    final_val_loader_info = final_train_loader_info # Use train as val

    # Final test loader info
    final_test_loader_info = test_loader_info.copy()
    final_test_loader_info['batch_size'] = batch_size
    final_test_loader_info['num_batches'] = int(np.ceil(final_test_loader_info['num_samples'] / batch_size))
    final_test_loader_info['use_one_hot'] = False

    regularization_types = ['none', 'l1', 'dropout', 'early_stopping'] # Removed 'l2'
    results = {}

    for reg in regularization_types:
        print(f"\nRunning final model with {reg} regularization")
        current_dropout_rate = dropout_rate if reg == 'dropout' else 0.0
        # Use 'softmax' output activation for Cross Entropy
        model = BasicMLP_NumPy(nHidden=nHidden, dropout_rate=current_dropout_rate, activation_fn='sigmoid', output_activation='softmax')

        # Use NumPy Cross Entropy loss and derivative
        loss_fn_ce, loss_deriv_fn_ce = cross_entropy_loss, cross_entropy_loss_derivative
        # Optimizer doesn't support weight_decay
        optimizer = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

        # Train using the CE version
        train_numpy_ce(model, final_train_loader_info, final_val_loader_info if reg == 'early_stopping' else None,
                       loss_fn_ce, loss_deriv_fn_ce, optimizer, num_epochs, regularization=reg, l1_lambda=l1_lambda if reg == 'l1' else 0.0, patience=5)

        # Test using the CE version
        test_accuracy = test_numpy_ce(model, final_test_loader_info)
        results[reg] = test_accuracy
        print(f'Final Test Accuracy with {reg}: {test_accuracy:.2f}%')

    # Create and print results table
    df = pd.DataFrame(list(results.items()), columns=['Regularization', 'Test Accuracy (%)'])
    print("\n=== Final Results Table (NumPy - Cross Entropy - L2 Excluded) ===")
    print(df)

# Train the final model with fixed hyperparameters using the CE version
train_final_model_numpy_ce()