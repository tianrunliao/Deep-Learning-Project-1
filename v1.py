import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import struct

# 设置随机种子以确保结果可重复
torch.manual_seed(321)

# 1. 自定义加载MNIST数据集
def load_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# 数据路径
train_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# 加载数据
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# 数据预处理：将图像归一化到[0, 1]并转换为PyTorch张量
train_images = torch.tensor(train_images, dtype=torch.float32) / 255.0
test_images = torch.tensor(test_images, dtype=torch.float32) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 将图像展平为1D向量（28x28 -> 784）
train_images = train_images.view(-1, 28 * 28)
test_images = test_images.view(-1, 28 * 28)

# 将标签转换为one-hot编码（用于MSE损失）
def to_one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

train_labels_one_hot = to_one_hot(train_labels)
test_labels_one_hot = to_one_hot(test_labels)

# 创建PyTorch数据集和数据加载器
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels_one_hot)
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels_one_hot)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 定义一个完全未优化的MLP模型（固定隐藏层结构）
class BasicMLP(nn.Module):
    def __init__(self):
        super(BasicMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),  # 输入层到第一个隐藏层，固定128个神经元
            nn.Sigmoid(),             # 使用简单的Sigmoid激活函数
            nn.Linear(128, 64),       # 第一个隐藏层到第二个隐藏层，固定64个神经元
            nn.Sigmoid(),
            nn.Linear(64, 10)         # 第二个隐藏层到输出层（10个类别）
        )

    def forward(self, x):
        return self.layers(x)

# 3. 初始化模型、损失函数和优化器
model = BasicMLP()  # 固定结构，不通过nHidden指定
criterion = nn.MSELoss()  # 使用均方误差损失（未使用交叉熵）
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 简单SGD，无动量

# 4. 训练模型
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# 5. 测试模型
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels, 1)  # 从one-hot转换回类别
            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 6. 运行训练和测试
print("Training the model...")
train(model, train_loader, criterion, optimizer, num_epochs=5)
print("\nTesting the model...")
test_accuracy = test(model, test_loader)