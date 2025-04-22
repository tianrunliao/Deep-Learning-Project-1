import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization

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

# 创建测试集的数据加载器
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels_one_hot)
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

# 3. 训练函数（每个fold单独训练）
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# 4. 测试函数
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
    return accuracy

# 5. 定义目标函数（供贝叶斯优化调用）
def objective(lr, num_epochs, batch_size, k_folds):
    # 确保超参数是合适的类型
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 初始化k折交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f'Fold {fold+1}/{k_folds}')

        # 划分训练和验证数据
        train_images_fold = train_images[train_idx]
        train_labels_fold = train_labels_one_hot[train_idx]
        val_images_fold = train_images[val_idx]
        val_labels_fold = train_labels_one_hot[val_idx]

        # 创建数据加载器
        train_dataset_fold = torch.utils.data.TensorDataset(train_images_fold, train_labels_fold)
        val_dataset_fold = torch.utils.data.TensorDataset(val_images_fold, val_labels_fold)

        train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        val_loader_fold = torch.utils.data.DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

        # 初始化模型、损失函数和优化器
        model = BasicMLP()
        criterion = nn.MSELoss()  # 使用均方误差损失
        optimizer = optim.SGD(model.parameters(), lr=lr)  # 简单SGD，无动量

        # 训练模型
        train(model, train_loader_fold, criterion, optimizer, num_epochs)

        # 在验证集上评估
        val_accuracy = test(model, val_loader_fold)
        fold_accuracies.append(val_accuracy)
        print(f'Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')

    # 返回平均验证准确率（贝叶斯优化需要最大化目标）
    avg_val_accuracy = np.mean(fold_accuracies)
    print(f'Average Validation Accuracy: {avg_val_accuracy:.2f}%')
    return avg_val_accuracy

# 6. 设置贝叶斯优化
pbounds = {
    'lr': (0.001, 0.1),         # 学习率范围
    'num_epochs': (5, 15),      # 训练轮数范围
    'batch_size': (16, 128),     # 批量大小范围
    'k_folds': (3, 5)            # 交叉验证折数范围
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

# 运行贝叶斯优化（初始化5个随机点，然后迭代10次）
print("Starting Bayesian Optimization...")
optimizer.maximize(
    init_points=5,  # 随机初始化的点数
    n_iter=10,      # 迭代次数
)

# 7. 打印最佳超参数
print("\n=== Best Hyperparameters ===")
best_params = optimizer.max
print(f"Best Average Validation Accuracy: {best_params['target']:.2f}%")
print(f"Best Parameters: {best_params['params']}")

# 8. 使用最佳超参数重新训练并测试
def train_final_model(best_params):
    print("\nTraining final model with best hyperparameters...")
    torch.manual_seed(42)

    # 提取最佳超参数
    lr = best_params['lr']
    num_epochs = int(best_params['num_epochs'])
    batch_size = int(best_params['batch_size'])

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels_one_hot)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    final_model = BasicMLP()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(final_model.parameters(), lr=lr)

    # 训练模型
    train(final_model, train_loader, criterion, optimizer, num_epochs)

    # 在测试集上评估
    print("\nTesting the final model on the test set...")
    test_accuracy = test(final_model, test_loader)
    print(f'Final Test Accuracy with best parameters: {test_accuracy:.2f}%')

# 9. 使用最佳超参数训练最终模型
train_final_model(best_params['params'])