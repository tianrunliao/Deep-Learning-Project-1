import numpy as np
import gzip
import struct
import os
# 移除 scipy，因为 NumPy 版 function.py 中没有用到它
# from scipy.ndimage import shift, rotate, zoom # NumPy 版不再需要

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

# 1. 加载 MNIST 数据 (使用 function.py)

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

# 2. 数据增强函数 (NumPy 版)
# 注意：NumPy 版本的数据增强可能与 PyTorch/SciPy 略有不同，且较慢
def augment_data_numpy(images_flat, labels):
    # 注意：输入是展平的图像 (N, 784)
    images_reshaped = images_flat.reshape(-1, 28, 28) # 先重塑回 2D
    augmented_images_list = []
    augmented_labels_list = []

    print("开始 NumPy 数据增强 (可能较慢)...")
    count = 0
    total = len(images_reshaped)

    for i in range(total):
        image = images_reshaped[i]
        label = labels[i]

        # 原始图像 (展平)
        augmented_images_list.append(image.flatten())
        augmented_labels_list.append(label)

        # 由于 NumPy 实现 shift, rotate, zoom 比较麻烦且可能引入新的依赖或复杂性
        # 我们在此简化，只添加原始图像。如果需要增强，建议使用专门的库（如 albumentations）或在数据加载前处理。
        # # 平移 (示例，可能需要 scipy)
        # try:
        #     from scipy.ndimage import shift
        #     shift_x, shift_y = np.random.randint(-2, 3, size=2)
        #     shifted_image = shift(image, [shift_x, shift_y], mode='nearest')
        #     augmented_images_list.append(shifted_image.flatten())
        #     augmented_labels_list.append(label)
        # except ImportError:
        #     pass # SciPy 不可用则跳过

        # # 旋转 (示例，可能需要 scipy)
        # try:
        #     from scipy.ndimage import rotate
        #     angle = np.random.uniform(-10, 10)
        #     rotated_image = rotate(image, angle, reshape=False, mode='nearest')
        #     augmented_images_list.append(rotated_image.flatten())
        #     augmented_labels_list.append(label)
        # except ImportError:
        #     pass # SciPy 不可用则跳过

        count += 1
        if count % 1000 == 0:
            print(f"  增强进度: {count}/{total}")

    print("NumPy 数据增强完成 (仅包含原始图像)。")
    return np.array(augmented_images_list), np.array(augmented_labels_list)

# 应用数据增强 (NumPy - 简化版，只返回原始数据)
print("应用数据增强 (NumPy 简化版)...")
# train_images 和 train_labels 已经是 NumPy 数组且已展平/归一化
augmented_train_images, augmented_train_labels = augment_data_numpy(train_images, train_labels)
print(f"原始训练样本: {len(train_images)}, 增强后训练样本: {len(augmented_train_images)}")

# 数据预处理：标签转换为one-hot编码 (NumPy)
num_classes = 10
augmented_train_labels_one_hot = one_hot_encode(augmented_train_labels, num_classes)
# 测试集标签也需要 one-hot
test_labels_one_hot = one_hot_encode(test_labels, num_classes)

# 创建数据加载器信息字典
batch_size = 64
train_loader_info = {
    'images': augmented_train_images,
    'labels': augmented_train_labels_one_hot,
    'batch_size': batch_size,
    'num_samples': augmented_train_images.shape[0],
    'num_batches': int(np.ceil(augmented_train_images.shape[0] / batch_size)),
    'shuffle': True,
    'use_one_hot': True,
    'num_classes': num_classes
}
test_loader_info = {
    'images': test_images, # 测试集不使用增强图像
    'labels': test_labels_one_hot,
    'batch_size': batch_size,
    'num_samples': test_images.shape[0],
    'num_batches': int(np.ceil(test_images.shape[0] / batch_size)),
    'shuffle': False,
    'use_one_hot': True,
    'num_classes': num_classes
}

# 3. 定义和初始化模型 (使用导入的 NumPy MLP)
#    结构 [128, 64]，激活 Sigmoid
hidden_layers = [128, 64]
activation = 'sigmoid'
model = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)

# 4. 初始化损失函数和优化器 (NumPy)
loss_fn, loss_derivative_fn = mse_loss, mse_loss_derivative
optimizer = SGD_Optimizer(model=model, learning_rate=0.1, momentum=0.0) # 调整学习率

# 5. 训练模型 (NumPy)
def train_numpy(model, loader_info, loss_fn, loss_derivative_fn, optimizer, num_epochs=5):
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
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{loader_info["num_batches"]}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
    model.set_training_mode(False)

# 6. 测试模型 (NumPy)
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
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 7. 运行训练和测试
print("Training the model (NumPy)...")
train_numpy(model, train_loader_info, loss_fn, loss_derivative_fn, optimizer, num_epochs=5)
print("\nTesting the model (NumPy)...")
test_accuracy = test_numpy(model, test_loader_info)