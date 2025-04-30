import numpy as np
import gzip
import struct
import os
from sklearn.model_selection import KFold

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

# 设置随机种子 (NumPy)
np.random.seed(321)

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

# 2. 定义模型结构 (NumPy)
hidden_layers = [128, 64]
activation = 'sigmoid'
# 模型实例将在 KFold 循环和最终训练中创建

# 3. 训练函数 (NumPy - MSE)
def train_numpy_mse(model, loader_info, loss_fn, loss_derivative_fn, optimizer, num_epochs=5):
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

# 4. 测试函数 (NumPy - MSE)
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
    # print(f'Test Accuracy: {accuracy:.2f}%') # 改为在调用处打印
    return accuracy

# 5. 实现5折交叉验证 (NumPy)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_accuracies = []

print("Starting 5-fold cross-validation (NumPy)...")
for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
    print(f'\nFold {fold+1}/{k}')

    # 划分训练和验证数据 (NumPy 索引)
    train_images_fold = train_images[train_idx]
    train_labels_fold = train_labels_one_hot[train_idx] # 使用 one-hot
    val_images_fold = train_images[val_idx]
    val_labels_fold = train_labels_one_hot[val_idx]   # 使用 one-hot

    # 创建当前 fold 的 loader info
    batch_size_fold = 64
    fold_train_loader_info = {
        'images': train_images_fold, 'labels': train_labels_fold,
        'batch_size': batch_size_fold,
        'num_samples': len(train_idx),
        'num_batches': int(np.ceil(len(train_idx) / batch_size_fold)),
        'shuffle': True, 'use_one_hot': True, 'num_classes': num_classes
    }
    fold_val_loader_info = {
        'images': val_images_fold, 'labels': val_labels_fold,
        'batch_size': batch_size_fold,
        'num_samples': len(val_idx),
        'num_batches': int(np.ceil(len(val_idx) / batch_size_fold)),
        'shuffle': False, 'use_one_hot': True, 'num_classes': num_classes
    }

    # 初始化模型、损失函数和优化器
    model_fold = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)
    loss_fn_mse, loss_derivative_fn_mse = mse_loss, mse_loss_derivative
    optimizer_fold = SGD_Optimizer(model=model_fold, learning_rate=0.1, momentum=0.0) # 调整学习率

    # 训练模型
    train_numpy_mse(model_fold, fold_train_loader_info, loss_fn_mse, loss_derivative_fn_mse, optimizer_fold, num_epochs=5)

    # 在验证集上评估
    print("Evaluating on validation set...")
    val_accuracy = test_numpy_mse(model_fold, fold_val_loader_info)
    fold_accuracies.append(val_accuracy)
    print(f'Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')

# 6. 计算交叉验证的平均准确率
avg_val_accuracy = np.mean(fold_accuracies)
print(f'\nAverage Validation Accuracy across {k} folds: {avg_val_accuracy:.2f}%')

# 7. 使用所有训练数据重新训练最终模型，并在测试集上评估 (NumPy)
print("\nTraining final model on all training data (NumPy)...")
final_model = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)
loss_fn_mse, loss_derivative_fn_mse = mse_loss, mse_loss_derivative
final_optimizer = SGD_Optimizer(model=final_model, learning_rate=0.1, momentum=0.0)

# 创建最终训练 loader info
final_train_loader_info = {
    'images': train_images, # 使用原始训练数据
    'labels': train_labels_one_hot,
    'batch_size': 64,
    'num_samples': train_images.shape[0],
    'num_batches': int(np.ceil(train_images.shape[0] / 64)),
    'shuffle': True,
    'use_one_hot': True,
    'num_classes': num_classes
}

train_numpy_mse(final_model, final_train_loader_info, loss_fn_mse, loss_derivative_fn_mse, final_optimizer, num_epochs=5)

print("\nTesting the final model on the test set (NumPy)...")
test_accuracy = test_numpy_mse(final_model, test_loader_info)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")