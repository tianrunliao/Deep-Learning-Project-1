import numpy as np
import gzip
import struct
import os # 新增
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization

# 导入 NumPy 版 function.py 组件
try:
    from function import (
        load_mnist_data, BasicMLP_NumPy,
        mse_loss, mse_loss_derivative, # MSE
        SGD_Optimizer,
        get_batch,
        one_hot_encode
    )
    print("成功从 function.py (NumPy 版) 导入模块。")
except ImportError as e:
    print(f"无法导入 function.py (NumPy 版): {e}")
    print("请确保 function.py 存在且已更新为 NumPy 版本。")
    exit()


# 1. 加载 MNIST 数据 (NumPy)
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

# 数据预处理 (NumPy)
# 图像已展平/归一化
# 标签 one-hot (用于 MSE)
num_classes = 10
train_labels_one_hot = one_hot_encode(train_labels, num_classes)
test_labels_one_hot = one_hot_encode(test_labels, num_classes)

# 测试集加载信息 (NumPy)
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

# 2. 定义模型结构 (NumPy) - 使用固定的结构，因为超参数优化不包括模型结构
hidden_layers = [128, 64]
activation = 'sigmoid'

# 3. 训练函数 (NumPy - MSE)
def train_numpy_mse(model, loader_info, loss_fn, loss_derivative_fn, optimizer, num_epochs):
    model.set_training_mode(True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        indices = np.arange(loader_info['num_samples'])
        if loader_info['shuffle']:
            np.random.shuffle(indices)
        for i in range(loader_info['num_batches']):
            batch_images, batch_labels_one_hot = get_batch(loader_info, i, indices)
            if batch_images is None: continue # 跳过空批次
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


# 4. 测试函数 (NumPy - MSE)
def test_numpy_mse(model, loader_info):
    model.set_training_mode(False)
    correct = 0
    total = 0
    for i in range(loader_info['num_batches']):
        images, labels_one_hot = get_batch(loader_info, i)
        if images is None: continue # 跳过空批次
        logits = model.forward(images)
        predicted_indices = np.argmax(logits, axis=1)
        true_indices = np.argmax(labels_one_hot, axis=1)
        total += true_indices.shape[0]
        correct += np.sum(predicted_indices == true_indices)
    if total == 0: return 0.0 # 避免除以零
    accuracy = 100 * correct / total
    return accuracy

# 5. 定义目标函数（供贝叶斯优化调用 - NumPy 版本）
def objective(lr, num_epochs, batch_size, k_folds):
    # 确保超参数是合适的类型
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    if k_folds < 2: k_folds = 2 # KFold 至少需要 2 折

    # 设置随机种子 (NumPy)
    np.random.seed(42)

    # 初始化k折交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    # 使用原始训练数据进行切分
    images_to_split = train_images
    labels_to_split = train_labels_one_hot

    for fold, (train_idx, val_idx) in enumerate(kf.split(images_to_split)):
        print(f'Fold {fold+1}/{k_folds}')

        # 划分训练和验证数据 (NumPy)
        train_images_fold = images_to_split[train_idx]
        train_labels_fold = labels_to_split[train_idx]
        val_images_fold = images_to_split[val_idx]
        val_labels_fold = labels_to_split[val_idx]

        # 创建当前 fold 的 loader info (NumPy)
        fold_train_loader_info = {
            'images': train_images_fold, 'labels': train_labels_fold,
            'batch_size': batch_size,
            'num_samples': len(train_idx),
            'num_batches': int(np.ceil(len(train_idx) / batch_size)),
            'shuffle': True, 'use_one_hot': True, 'num_classes': num_classes
        }
        fold_val_loader_info = {
            'images': val_images_fold, 'labels': val_labels_fold,
            'batch_size': batch_size, # 验证集也用同样的 batch_size
            'num_samples': len(val_idx),
            'num_batches': int(np.ceil(len(val_idx) / batch_size)),
            'shuffle': False, 'use_one_hot': True, 'num_classes': num_classes
        }

        # 初始化模型、损失函数和优化器 (NumPy)
        model_fold = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)
        loss_fn_mse, loss_derivative_fn_mse = mse_loss, mse_loss_derivative
        optimizer_fold = SGD_Optimizer(model=model_fold, learning_rate=lr, momentum=0.0) # 无动量

        # 训练模型 (NumPy)
        train_numpy_mse(model_fold, fold_train_loader_info, loss_fn_mse, loss_derivative_fn_mse, optimizer_fold, num_epochs)

        # 在验证集上评估 (NumPy)
        val_accuracy = test_numpy_mse(model_fold, fold_val_loader_info)
        fold_accuracies.append(val_accuracy)
        print(f'Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')

    # 返回平均验证准确率（贝叶斯优化需要最大化目标）
    avg_val_accuracy = np.mean(fold_accuracies)
    print(f'Average Validation Accuracy for params (lr={lr:.4f}, epochs={num_epochs}, batch={batch_size}, folds={k_folds}): {avg_val_accuracy:.2f}%')
    return avg_val_accuracy

# 6. 设置贝叶斯优化
# 调整 k_folds 的范围，确保至少为 2
pbounds = {
    'lr': (0.001, 0.1),
    'num_epochs': (5, 15),
    'batch_size': (16, 128),
    'k_folds': (2, 5) # 确保 k_folds >= 2
}

optimizer_bo = BayesianOptimization( # 重命名避免与 SGD_Optimizer 冲突
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

# 运行贝叶斯优化（初始化5个随机点，然后迭代10次）
print("Starting Bayesian Optimization (NumPy)...")
optimizer_bo.maximize(
    init_points=5,  # 随机初始化的点数
    n_iter=10,      # 迭代次数
)

# 7. 打印最佳超参数
print("\n=== Best Hyperparameters (NumPy) ===")
best_params = optimizer_bo.max
print(f"Best Average Validation Accuracy: {best_params['target']:.2f}%")
# 显式打印最佳参数，确保类型正确
best_lr = best_params['params']['lr']
best_num_epochs = int(best_params['params']['num_epochs'])
best_batch_size = int(best_params['params']['batch_size'])
best_k_folds = int(best_params['params']['k_folds']) # 虽然最终训练不用 k_folds，但打印出来
print(f"Best Parameters: lr={best_lr:.4f}, num_epochs={best_num_epochs}, batch_size={best_batch_size}, k_folds={best_k_folds}")


# 8. 使用最佳超参数重新训练并测试 (NumPy)
def train_final_model_numpy(best_params_dict):
    print("\nTraining final model with best hyperparameters (NumPy)...")
    np.random.seed(42) # 设置 NumPy 随机种子

    # 提取最佳超参数
    lr = best_params_dict['lr']
    num_epochs = int(best_params_dict['num_epochs'])
    batch_size = int(best_params_dict['batch_size'])

    # 创建最终训练 loader info (NumPy)
    final_train_loader_info = {
        'images': train_images, # 使用全部训练数据
        'labels': train_labels_one_hot,
        'batch_size': batch_size,
        'num_samples': train_images.shape[0],
        'num_batches': int(np.ceil(train_images.shape[0] / batch_size)),
        'shuffle': True, 'use_one_hot': True, 'num_classes': num_classes
    }

    # 初始化模型、损失函数和优化器 (NumPy)
    final_model = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)
    loss_fn_mse, loss_derivative_fn_mse = mse_loss, mse_loss_derivative
    final_optimizer = SGD_Optimizer(model=final_model, learning_rate=lr, momentum=0.0)

    # 训练模型 (NumPy)
    train_numpy_mse(final_model, final_train_loader_info, loss_fn_mse, loss_derivative_fn_mse, final_optimizer, num_epochs)

    # 在测试集上评估 (NumPy) - 使用之前定义的 test_loader_info
    print("\nTesting the final model on the test set (NumPy)...")
    test_accuracy = test_numpy_mse(final_model, test_loader_info)
    print(f'Final Test Accuracy with best parameters: {test_accuracy:.2f}%')

# 9. 使用最佳超参数训练最终模型 (NumPy)
train_final_model_numpy(best_params['params'])