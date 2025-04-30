import numpy as np
import gzip
import struct
import os
from sklearn.model_selection import KFold
# 移除 bayes_opt 和 scipy
# from bayes_opt import BayesianOptimization
# from scipy.ndimage import shift, rotate, zoom

# 导入 NumPy 版 function.py 组件
try:
    from function import (
        load_mnist_data, BasicMLP_NumPy,
        mse_loss, mse_loss_derivative, # MSE 损失及其梯度
        SGD_Optimizer, # NumPy SGD 优化器
        get_batch, # 获取批次数据
        one_hot_encode # NumPy one-hot 编码
        # 交叉熵不需要
    )
    print("成功从 function.py (NumPy 版) 导入模块。")
except ImportError as e:
    print(f"无法导入 function.py (NumPy 版): {e}")
    print("请确保 function.py 存在且已更新为 NumPy 版本。")
    exit()

# 移除 torch

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
# 同样，简化数据增强，只包含原始数据
def augment_data_numpy_v3(images_flat, labels):
    print("应用 NumPy 数据增强 (简化版，仅原始数据)...")
    return images_flat, labels

# 应用数据增强 (简化)
augmented_train_images, augmented_train_labels = augment_data_numpy_v3(train_images, train_labels)
combined_train_images = augmented_train_images
combined_train_labels = augmented_train_labels
print(f"训练样本数量: {len(combined_train_images)}")

# ---------------------------
# 3. 数据预处理 (NumPy)
# ---------------------------
# 图像已由 load_mnist_data 展平和归一化
# 标签转换为 one-hot (因为后面用了 MSE)
num_classes = 10
combined_train_labels_one_hot = one_hot_encode(combined_train_labels, num_classes)
test_labels_one_hot = one_hot_encode(test_labels, num_classes)

# 准备测试集加载信息
batch_size_test = 64 # 固定测试批量大小
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
# 4. 定义 MLP 模型 (使用 NumPy 版 function.py)
# ---------------------------
# 模型结构固定，激活函数 Sigmoid
hidden_layers = [128, 64]
activation = 'sigmoid'
# 模型将在 KFold 循环和最终训练中实例化

# ---------------------------
# 5. 定义训练和测试函数 (NumPy - MSE)
# ---------------------------
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
    # print(f'Accuracy: {accuracy:.2f}%') # test 函数通常只返回准确率
    return accuracy

# ---------------------------
# 6. 定义目标函数 (NumPy 版本)
# ---------------------------
# 注意：贝叶斯优化库 bayes-opt 可能与纯 NumPy 不兼容
# 这里我们定义函数结构，但实际运行贝叶斯优化需要该库
def objective_numpy(lr, num_epochs, batch_size, k_folds):
    print(f"\n--- Testing config: lr={lr:.4f}, epochs={int(num_epochs)}, batch={int(batch_size)}, folds={int(k_folds)} ---")
    # 将超参数转为整型
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)

    # 设置随机种子确保可复现性
    np.random.seed(42)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    all_train_images_cv = combined_train_images
    all_train_labels_cv = combined_train_labels_one_hot # MSE 需要 one-hot

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_images_cv)):
        print(f'  Fold {fold+1}/{k_folds}')

        train_images_fold = all_train_images_cv[train_idx]
        train_labels_fold = all_train_labels_cv[train_idx]
        val_images_fold = all_train_images_cv[val_idx]
        val_labels_fold = all_train_labels_cv[val_idx]

        fold_train_loader_info = {
            'images': train_images_fold,
            'labels': train_labels_fold,
            'batch_size': batch_size,
            'num_samples': len(train_idx),
            'num_batches': int(np.ceil(len(train_idx) / batch_size)),
            'shuffle': True,
            'use_one_hot': True,
            'num_classes': num_classes
        }
        fold_val_loader_info = {
            'images': val_images_fold,
            'labels': val_labels_fold,
            'batch_size': batch_size,
            'num_samples': len(val_idx),
            'num_batches': int(np.ceil(len(val_idx) / batch_size)),
            'shuffle': False,
            'use_one_hot': True,
            'num_classes': num_classes
        }

        model_fold = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)
        loss_fn_mse, loss_derivative_fn_mse = mse_loss, mse_loss_derivative
        optimizer_fold = SGD_Optimizer(model=model_fold, learning_rate=lr, momentum=0.0) # 使用传入的 lr

        train_numpy_mse(model_fold, fold_train_loader_info, loss_fn_mse, loss_derivative_fn_mse, optimizer_fold, num_epochs)

        val_accuracy = test_numpy_mse(model_fold, fold_val_loader_info)
        fold_accuracies.append(val_accuracy)
        print(f'  Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')

    avg_val_accuracy = np.mean(fold_accuracies)
    print(f'--- Config Avg Accuracy: {avg_val_accuracy:.2f}% --- ')
    return avg_val_accuracy

# ---------------------------
# 7. 贝叶斯优化 (需要 bayes-opt 库)
# ---------------------------
# 尝试导入 bayes_opt
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
        'lr': (0.01, 0.2),         # 调整学习率范围 (NumPy 可能需要更高)
        'num_epochs': (5, 15),
        'batch_size': (32, 128),
        'k_folds': (3, 5)
    }

    bayes_optimizer = BayesianOptimization(
        f=objective_numpy, # 使用 NumPy 目标函数
        pbounds=pbounds,
        random_state=42,
        verbose=2 # verbose = 2 时打印优化过程信息
    )

    print("\nStarting Bayesian Optimization (NumPy)...")
    try:
        bayes_optimizer.maximize(
            init_points=5, # 初始随机探索点数
            n_iter=10,     # 优化迭代次数
        )
        # 获取最佳参数
        best_params = bayes_optimizer.max['params']
        print("\n=== Best Hyperparameters Found ===")
        print(f"Best Average Validation Accuracy: {bayes_optimizer.max['target']:.2f}%")
        print(f"Best Parameters: {best_params}")
    except Exception as e:
        print(f"\n贝叶斯优化过程中出错: {e}")
        print("将使用默认参数训练最终模型。")
        best_params = None

# ---------------------------
# 8. 训练最终模型 (NumPy)
# ---------------------------
def train_final_model_numpy(params):
    print("\nTraining final model (NumPy)...")
    np.random.seed(42) # 确保最终训练也可复现

    # 使用找到的最佳参数或默认值
    if params:
        print("Using best hyperparameters found by Bayesian Optimization.")
        lr = params['lr']
        num_epochs = int(params['num_epochs'])
        batch_size = int(params['batch_size'])
    else:
        print("Using default hyperparameters.")
        lr = 0.1 # 默认 NumPy 学习率
        num_epochs = 10
        batch_size = 64

    # 创建最终训练数据加载信息 (使用所有合并数据)
    final_train_loader_info = {
        'images': combined_train_images,
        'labels': combined_train_labels_one_hot,
        'batch_size': batch_size,
        'num_samples': combined_train_images.shape[0],
        'num_batches': int(np.ceil(combined_train_images.shape[0] / batch_size)),
        'shuffle': True,
        'use_one_hot': True,
        'num_classes': num_classes
    }

    # 初始化最终模型、损失和优化器
    final_model = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)
    loss_fn_mse, loss_derivative_fn_mse = mse_loss, mse_loss_derivative
    final_optimizer = SGD_Optimizer(model=final_model, learning_rate=lr, momentum=0.0)

    # 训练最终模型
    train_numpy_mse(final_model, final_train_loader_info, loss_fn_mse, loss_derivative_fn_mse, final_optimizer, num_epochs)

    # 在测试集上评估最终模型
    print("\nTesting the final model on the test set...")
    final_test_accuracy = test_numpy_mse(final_model, test_loader_info)
    print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")

# 运行最终模型训练
train_final_model_numpy(best_params)