import numpy as np
import gzip
import struct
import os
from sklearn.model_selection import KFold

# 导入 NumPy 版 function.py 组件
try:
    from function import (
        load_mnist_data, BasicMLP_NumPy,
        mse_loss, mse_loss_derivative, # MSE 损失及其梯度
        SGD_Optimizer, # NumPy SGD 优化器
        get_batch, # 获取批次数据
        one_hot_encode # NumPy one-hot 编码
    )
    print("成功从 function.py (NumPy 版) 导入模块。")
except ImportError as e:
    print(f"无法导入 function.py (NumPy 版): {e}")
    print("请确保 function.py 存在且已更新为 NumPy 版本。")
    exit()

# 设置随机种子 (NumPy)
np.random.seed(321)

# ---------------------------
# 1. 数据加载 (NumPy)
# ---------------------------
# 数据路径
script_dir = os.path.dirname(os.path.abspath(__file__))
default_data_dir = os.path.join(script_dir, 'dataset', 'MNIST')
if not os.path.exists(default_data_dir):
    default_data_dir = os.path.join(os.path.dirname(script_dir), 'dataset', 'MNIST')
data_dir = default_data_dir
print(f"使用 MNIST 数据目录: {data_dir}")

try:
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    train_images, train_labels = load_mnist_data(train_images_path, train_labels_path)

    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    test_images, test_labels = load_mnist_data(test_images_path, test_labels_path)
except FileNotFoundError:
    print(f"错误：在 '{data_dir}' 中找不到 MNIST 数据文件。请检查路径。")
    exit()

# ---------------------------
# 2. 数据增强 (NumPy - 简化版)
# ---------------------------
# 简化增强，只包含原始图像及其标签
def augment_data_numpy_v4(images_flat, labels):
    print("应用 NumPy 数据增强 (简化版，仅原始数据)...")
    # 此版本原始代码增强后是4倍数据量，简化版只返回1倍
    return images_flat, labels

# 应用数据增强 (简化)
train_images_augmented, train_labels_augmented = augment_data_numpy_v4(train_images, train_labels)
print(f"训练样本数量: {len(train_images_augmented)}")

# ---------------------------
# 3. 数据预处理 (NumPy)
# ---------------------------
# 图像已展平/归一化
# 标签 one-hot (用于 MSE)
num_classes = 10
train_labels_one_hot = one_hot_encode(train_labels_augmented, num_classes)
test_labels_one_hot = one_hot_encode(test_labels, num_classes)

# 测试集加载信息
batch_size_test = 64
test_loader_info = {
    'images': test_images,
    'labels': test_labels_one_hot,
    'batch_size': batch_size_test,
    'num_samples': test_images.shape[0],
    'num_batches': int(np.ceil(test_images.shape[0] / batch_size_test)),
    'shuffle': False,
    'use_one_hot': True,
    'num_classes': num_classes
}

# ---------------------------
# 4. 模型定义 (NumPy)
# ---------------------------
# MLP 模型将在 objective 函数和最终训练中根据 nHidden_candidates 创建
activation = 'sigmoid' # 固定激活函数
nHidden_candidates = [
    [128],
    [256, 128],
    [512, 256, 128]
]

# ---------------------------
# 5. 训练和测试函数 (NumPy - MSE)
# ---------------------------
# (复用 pro_v3.py 中定义的 train_numpy_mse 和 test_numpy_mse)
def train_numpy_mse(model, loader_info, loss_fn, loss_derivative_fn, optimizer, num_epochs):
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
            grad_loss = loss_derivative_fn(logits, batch_labels_one_hot)
            model.backward(grad_loss)
            optimizer.step()
            running_loss += loss
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{loader_info["num_batches"]}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    model.set_training_mode(False)

def test_numpy_mse(model, loader_info):
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

# ---------------------------
# 6. 目标函数 (NumPy - 包含 nHidden)
# ---------------------------
def objective_numpy_v4(lr, num_epochs, batch_size, k_folds, nHidden_index):
    print(f"\n--- Testing config: lr={lr:.4f}, epochs={int(num_epochs)}, batch={int(batch_size)}, folds={int(k_folds)}, nHidden={nHidden_candidates[int(nHidden_index)]} ---")
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    nHidden_index = int(np.round(nHidden_index)) # BayesOpt 可能产生浮点数索引
    if not 0 <= nHidden_index < len(nHidden_candidates):
         print(f"警告: 无效的 nHidden_index ({nHidden_index}), 使用默认索引 0。")
         nHidden_index = 0
    nHidden = nHidden_candidates[nHidden_index]

    np.random.seed(42)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    all_train_images_cv = train_images_augmented
    all_train_labels_cv = train_labels_one_hot

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_images_cv)):
        print(f'  Fold {fold+1}/{k_folds}')
        train_images_fold = all_train_images_cv[train_idx]
        train_labels_fold = all_train_labels_cv[train_idx]
        val_images_fold = all_train_images_cv[val_idx]
        val_labels_fold = all_train_labels_cv[val_idx]

        fold_train_loader_info = {
            'images': train_images_fold, 'labels': train_labels_fold,
            'batch_size': batch_size, 'num_samples': len(train_idx),
            'num_batches': int(np.ceil(len(train_idx) / batch_size)),
            'shuffle': True, 'use_one_hot': True, 'num_classes': num_classes
        }
        fold_val_loader_info = {
            'images': val_images_fold, 'labels': val_labels_fold,
            'batch_size': batch_size, 'num_samples': len(val_idx),
            'num_batches': int(np.ceil(len(val_idx) / batch_size)),
            'shuffle': False, 'use_one_hot': True, 'num_classes': num_classes
        }

        # 使用选定的 nHidden 创建模型
        model_fold = BasicMLP_NumPy(nHidden=nHidden, activation_fn=activation, dropout_rate=0.0)
        loss_fn_mse, loss_derivative_fn_mse = mse_loss, mse_loss_derivative
        optimizer_fold = SGD_Optimizer(model=model_fold, learning_rate=lr, momentum=0.0)

        train_numpy_mse(model_fold, fold_train_loader_info, loss_fn_mse, loss_derivative_fn_mse, optimizer_fold, num_epochs)
        val_accuracy = test_numpy_mse(model_fold, fold_val_loader_info)
        fold_accuracies.append(val_accuracy)
        print(f'  Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')

    avg_val_accuracy = np.mean(fold_accuracies)
    print(f'--- Config Avg Accuracy: {avg_val_accuracy:.2f}% --- ')
    return avg_val_accuracy

# ---------------------------
# 7. 贝叶斯优化设置 (NumPy)
# ---------------------------
try:
    from bayes_opt import BayesianOptimization
    bayes_opt_available = True
except ImportError:
    print("\n警告: 贝叶斯优化库 'bayes_opt' 未安装。")
    print("将跳过超参数优化，并使用默认参数训练最终模型。")
    bayes_opt_available = False

best_params = None

if bayes_opt_available:
    pbounds = {
        'lr': (0.01, 0.2),
        'num_epochs': (5, 15),
        'batch_size': (32, 128),
        'k_folds': (3, 5),
        # nHidden_index 需要是离散的，但 bayes_opt 通常处理连续值
        # 我们传入索引范围，然后在 objective 函数中取整
        'nHidden_index': (0, len(nHidden_candidates) - 0.01) # 略小于上限以防取整问题
    }

    bayes_optimizer = BayesianOptimization(
        f=objective_numpy_v4, # 使用新的 NumPy 目标函数
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    print("\nStarting Bayesian Optimization (NumPy with nHidden)...")
    try:
        bayes_optimizer.maximize(init_points=5, n_iter=10)
        best_params = bayes_optimizer.max['params']
        print("\n=== Best Hyperparameters Found ===")
        print(f"Best Average Validation Accuracy: {bayes_optimizer.max['target']:.2f}%")
        print(f"Best Parameters: {best_params}")
    except Exception as e:
        print(f"\n贝叶斯优化过程中出错: {e}")
        print("将使用默认参数训练最终模型。")
        best_params = None
else:
     best_params = None # 确保未执行优化时 best_params 为 None

# ---------------------------
# 8. 训练最终模型 (NumPy)
# ---------------------------
def train_final_model_numpy_v4(params):
    print("\nTraining final model (NumPy)...")
    np.random.seed(42)

    if params:
        print("Using best hyperparameters found by Bayesian Optimization.")
        lr = params['lr']
        num_epochs = int(params['num_epochs'])
        batch_size = int(params['batch_size'])
        nHidden_index = int(np.round(params['nHidden_index']))
        if not 0 <= nHidden_index < len(nHidden_candidates):
             print(f"警告: 修正无效的 nHidden_index ({nHidden_index}) 为 0。")
             nHidden_index = 0
        nHidden = nHidden_candidates[nHidden_index]
    else:
        print("Using default hyperparameters.")
        lr = 0.1
        num_epochs = 10
        batch_size = 64
        nHidden = nHidden_candidates[1] # 默认使用 medium 结构

    print(f"Selected nHidden configuration: {nHidden}")

    final_train_loader_info = {
        'images': train_images_augmented,
        'labels': train_labels_one_hot,
        'batch_size': batch_size,
        'num_samples': train_images_augmented.shape[0],
        'num_batches': int(np.ceil(train_images_augmented.shape[0] / batch_size)),
        'shuffle': True,
        'use_one_hot': True,
        'num_classes': num_classes
    }

    final_model = BasicMLP_NumPy(nHidden=nHidden, activation_fn=activation, dropout_rate=0.0)
    loss_fn_mse, loss_derivative_fn_mse = mse_loss, mse_loss_derivative
    final_optimizer = SGD_Optimizer(model=final_model, learning_rate=lr, momentum=0.0)

    train_numpy_mse(final_model, final_train_loader_info, loss_fn_mse, loss_derivative_fn_mse, final_optimizer, num_epochs)

    print("\nTesting the final model on the test set...")
    final_test_accuracy = test_numpy_mse(final_model, test_loader_info)
    print(f'Final Test Accuracy: {final_test_accuracy:.2f}%')

# 运行最终模型训练
train_final_model_numpy_v4(best_params)