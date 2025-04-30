import numpy as np
import gzip
import struct
import os
from sklearn.model_selection import KFold

# 导入 NumPy 版 function.py 组件
try:
    from function import (
        load_mnist_data, BasicMLP_NumPy,
        cross_entropy_loss, cross_entropy_loss_derivative, softmax, # 交叉熵相关
        SGD_Optimizer, # NumPy SGD 优化器
        get_batch # 获取批次数据
        # one_hot_encode 不需要，因为用交叉熵
    )
    print("成功从 function.py (NumPy 版) 导入模块。")
except ImportError as e:
    print(f"无法导入 function.py (NumPy 版): {e}")
    print("请确保 function.py 存在且已更新为 NumPy 版本。")
    exit()

# 设置随机种子 (NumPy)
np.random.seed(321)

####################################
# 1. 数据加载与预处理 (NumPy)
####################################

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

# train_images, test_images 已经是 (N, 784) 的 NumPy float32 数组
# train_labels, test_labels 是 (N,) 的 NumPy uint8 数组 (类别索引)

####################################
# 2. 数据增强 (NumPy - 简化版)
####################################

# 同样，简化数据增强，只包含原始数据
def augment_data_numpy_v2(images_flat, labels):
    print("应用 NumPy 数据增强 (简化版，仅原始数据)...")
    # 直接返回原始数据即可，因为 function.py 的 load_mnist_data 已展平
    return images_flat, labels

# 应用数据增强 (简化)
augmented_train_images, augmented_train_labels = augment_data_numpy_v2(train_images, train_labels)
# 在这个简化版本中，它们与原始数据相同
combined_train_images = augmented_train_images
combined_train_labels = augmented_train_labels
print(f"训练样本数量: {len(combined_train_images)}")

####################################
# 3. 准备数据加载信息 (NumPy)
####################################

# 不需要 one-hot 编码
batch_size = 64
num_classes = 10

train_loader_info = {
    'images': combined_train_images,
    'labels': combined_train_labels, # 使用索引标签
    'batch_size': batch_size,
    'num_samples': combined_train_images.shape[0],
    'num_batches': int(np.ceil(combined_train_images.shape[0] / batch_size)),
    'shuffle': True,
    'use_one_hot': False, # 标签是索引
    'num_classes': num_classes
}

test_loader_info = {
    'images': test_images,
    'labels': test_labels, # 使用索引标签
    'batch_size': batch_size,
    'num_samples': test_images.shape[0],
    'num_batches': int(np.ceil(test_images.shape[0] / batch_size)),
    'shuffle': False,
    'use_one_hot': False, # 标签是索引
    'num_classes': num_classes
}

####################################
# 4. 定义 MLP 模型 (使用 NumPy)
####################################

# 使用导入的 BasicMLP_NumPy
hidden_layers = [128, 64]
activation = 'sigmoid'
# 初始化模型在交叉验证循环内部进行

####################################
# 5. 定义训练和测试函数 (NumPy 版本 - 交叉熵)
####################################

def train_numpy_ce(model, loader_info, loss_fn, loss_derivative_fn, optimizer, num_epochs=5):
    model.set_training_mode(True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        indices = np.arange(loader_info['num_samples'])
        if loader_info['shuffle']:
            np.random.shuffle(indices)

        for i in range(loader_info['num_batches']):
            batch_images, batch_labels_indices = get_batch(loader_info, i, indices)

            # 前向传播
            logits = model.forward(batch_images)
            # 计算交叉熵损失 (直接使用 logits)
            loss = loss_fn(logits, batch_labels_indices)
            # 计算梯度 (直接使用 logits - 假设导数函数也期望 logits)
            grad_loss = loss_derivative_fn(logits, batch_labels_indices)

            # 反向传播
            model.backward(grad_loss)
            # 更新参数
            optimizer.step()

            running_loss += loss
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{loader_info["num_batches"]}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    model.set_training_mode(False)

def test_numpy_ce(model, loader_info):
    model.set_training_mode(False)
    correct = 0
    total = 0
    for i in range(loader_info['num_batches']):
        images, true_labels_indices = get_batch(loader_info, i)

        logits = model.forward(images)
        predicted_indices = np.argmax(logits, axis=1)

        total += true_labels_indices.shape[0]
        correct += np.sum(predicted_indices == true_labels_indices)

    accuracy = 100 * correct / total
    return accuracy

####################################
# 6. 5折交叉验证 (NumPy)
####################################

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_accuracies = []

print("Starting 5-fold cross-validation (NumPy)...")

# 使用完整的训练数据进行分割
all_train_images = combined_train_images
all_train_labels = combined_train_labels # 索引标签

for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_images)):
    print(f'\nFold {fold+1}/{k}')

    # 划分训练和验证数据 (NumPy 索引)
    train_images_fold = all_train_images[train_idx]
    train_labels_fold = all_train_labels[train_idx]
    val_images_fold = all_train_images[val_idx]
    val_labels_fold = all_train_labels[val_idx]

    # 创建当前 fold 的 loader info
    fold_train_loader_info = {
        'images': train_images_fold,
        'labels': train_labels_fold,
        'batch_size': batch_size,
        'num_samples': len(train_idx),
        'num_batches': int(np.ceil(len(train_idx) / batch_size)),
        'shuffle': True,
        'use_one_hot': False,
        'num_classes': num_classes
    }
    fold_val_loader_info = {
        'images': val_images_fold,
        'labels': val_labels_fold,
        'batch_size': batch_size,
        'num_samples': len(val_idx),
        'num_batches': int(np.ceil(len(val_idx) / batch_size)),
        'shuffle': False,
        'use_one_hot': False,
        'num_classes': num_classes
    }

    # 初始化模型、损失函数和优化器
    model_fold = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)
    loss_fn_ce, loss_derivative_fn_ce = cross_entropy_loss, cross_entropy_loss_derivative
    optimizer_fold = SGD_Optimizer(model=model_fold, learning_rate=0.1, momentum=0.0)

    # 训练当前 fold 模型
    train_numpy_ce(model_fold, fold_train_loader_info, loss_fn_ce, loss_derivative_fn_ce, optimizer_fold, num_epochs=5)

    # 在验证集上评估
    print("Evaluating on validation set...")
    val_accuracy = test_numpy_ce(model_fold, fold_val_loader_info)
    fold_accuracies.append(val_accuracy)
    # 打印验证准确率时不需要再次调用 test 函数
    print(f'Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')

# 计算交叉验证的平均准确率
avg_val_accuracy = np.mean(fold_accuracies)
print(f'\nAverage Validation Accuracy across {k} folds: {avg_val_accuracy:.2f}%')

####################################
# 7. 使用所有训练数据重新训练最终模型，并在测试集上评估 (NumPy)
####################################

print("\nTraining final model on all training data (NumPy)...")
final_model = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)
loss_fn_ce, loss_derivative_fn_ce = cross_entropy_loss, cross_entropy_loss_derivative
final_optimizer = SGD_Optimizer(model=final_model, learning_rate=0.1, momentum=0.0)

# 使用包含所有合并数据的 train_loader_info
train_numpy_ce(final_model, train_loader_info, loss_fn_ce, loss_derivative_fn_ce, final_optimizer, num_epochs=5)

print("\nTesting the final model on the test set (NumPy)...")
test_accuracy = test_numpy_ce(final_model, test_loader_info)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")