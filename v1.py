import numpy as np
import gzip
import struct
import os # 用于数据路径

# 从 NumPy 版本的 function.py 导入所需组件
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

# 设置随机种子以确保结果可重复 (NumPy 版本)
np.random.seed(321)

# 1. 加载 MNIST 数据集 (使用 function.py 中的 NumPy 函数)
#    不再需要自定义的 load_mnist_images/labels

# 数据路径 (尝试使其更通用)
script_dir = os.path.dirname(os.path.abspath(__file__))
default_data_dir = os.path.join(script_dir, 'dataset', 'MNIST')
# 如果默认路径不存在，尝试上级目录
if not os.path.exists(default_data_dir):
    default_data_dir = os.path.join(os.path.dirname(script_dir), 'dataset', 'MNIST')

data_dir = default_data_dir # 可以修改为实际路径
print(f"使用 MNIST 数据目录: {data_dir}")

try:
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    train_images, train_labels = load_mnist_data(train_images_path, train_labels_path) # 已展平并归一化

    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    test_images, test_labels = load_mnist_data(test_images_path, test_labels_path) # 已展平并归一化
except FileNotFoundError:
    print(f"错误：在 '{data_dir}' 中找不到 MNIST 数据文件。请检查路径。")
    exit()

# 数据预处理：标签转换为 one-hot 编码 (NumPy 版本)
num_classes = 10
train_labels_one_hot = one_hot_encode(train_labels, num_classes)
test_labels_one_hot = one_hot_encode(test_labels, num_classes)

# 创建数据加载器信息字典 (替代 DataLoader)
batch_size = 64
train_loader_info = {
    'images': train_images,
    'labels': train_labels_one_hot, # 使用 one-hot 标签
    'batch_size': batch_size,
    'num_samples': train_images.shape[0],
    'num_batches': int(np.ceil(train_images.shape[0] / batch_size)),
    'shuffle': True,
    'use_one_hot': True, # 表明标签已是 one-hot
    'num_classes': num_classes
}
test_loader_info = {
    'images': test_images,
    'labels': test_labels_one_hot, # 使用 one-hot 标签
    'batch_size': batch_size,
    'num_samples': test_images.shape[0],
    'num_batches': int(np.ceil(test_images.shape[0] / batch_size)),
    'shuffle': False,
    'use_one_hot': True, # 表明标签已是 one-hot
    'num_classes': num_classes
}

# 2. 定义和初始化模型 (使用导入的 NumPy MLP)
#    固定结构 [128, 64]，激活函数 Sigmoid
hidden_layers = [128, 64]
activation = 'sigmoid'
model = BasicMLP_NumPy(nHidden=hidden_layers, activation_fn=activation, dropout_rate=0.0)

# 3. 初始化损失函数和优化器 (NumPy 版本)
#    使用 MSE 损失
loss_fn, loss_derivative_fn = mse_loss, mse_loss_derivative
optimizer = SGD_Optimizer(model=model, learning_rate=0.1, momentum=0.0) # 调整学习率，无动量

# 4. 训练模型 (NumPy 版本)
def train_numpy(model, loader_info, loss_fn, loss_derivative_fn, optimizer, num_epochs=5):
    model.set_training_mode(True) # 训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        indices = np.arange(loader_info['num_samples'])
        if loader_info['shuffle']:
            np.random.shuffle(indices)

        for i in range(loader_info['num_batches']):
            # 获取批次数据 (已经是 one-hot)
            batch_images, batch_labels_one_hot = get_batch(loader_info, i, indices)

            # 前向传播
            logits = model.forward(batch_images) # 输出是 logits

            # 计算损失 (MSE)
            loss = loss_fn(logits, batch_labels_one_hot)

            # 计算梯度 (MSE)
            grad_loss = loss_derivative_fn(logits, batch_labels_one_hot)

            # 反向传播
            model.backward(grad_loss)

            # 更新参数
            optimizer.step()

            running_loss += loss
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{loader_info["num_batches"]}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    model.set_training_mode(False) # 结束时设为评估模式

# 5. 测试模型 (NumPy 版本)
def test_numpy(model, loader_info):
    model.set_training_mode(False) # 评估模式
    correct = 0
    total = 0
    for i in range(loader_info['num_batches']):
        images, labels_one_hot = get_batch(loader_info, i)

        logits = model.forward(images)
        predicted_indices = np.argmax(logits, axis=1)
        true_indices = np.argmax(labels_one_hot, axis=1) # 从 one-hot 转回索引

        total += true_indices.shape[0]
        correct += np.sum(predicted_indices == true_indices)

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 6. 运行训练和测试
print("Training the model (NumPy)...")
train_numpy(model, train_loader_info, loss_fn, loss_derivative_fn, optimizer, num_epochs=5)
print("Testing the model (NumPy)...")
test_accuracy = test_numpy(model, test_loader_info)