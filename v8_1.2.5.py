import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import pandas as pd

# 设置使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载 MNIST 数据集
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

# 加载和预处理数据
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# 归一化并转换为张量（注意转为 float 同时除以 255.0）
train_images = torch.tensor(train_images, dtype=torch.float32) / 255.0
test_images = torch.tensor(test_images, dtype=torch.float32) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 添加通道维度，调整为 CNN 输入格式： [N, 1, 28, 28]
train_images = train_images.unsqueeze(1)
test_images = test_images.unsqueeze(1)

# 创建测试数据集和 DataLoader
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义具有可配置卷积层和 dropout 的 CNN 模型
class BasicCNN(nn.Module):
    def __init__(self, conv_filters, dropout_rate=0.0):
        super(BasicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels=1, out_channels=conv_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: [N, conv_filters[0], 14, 14]
            
            # 第二层卷积
            nn.Conv2d(in_channels=conv_filters[0], out_channels=conv_filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 输出: [N, conv_filters[1], 7, 7]
        )
        
        # 展平后特征的维度
        self.flatten_dim = conv_filters[1] * 7 * 7
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(128, 10)  # 输出：10 类
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x  # 返回原始 logits

# 训练函数（去除 L1 正则化相关部分）
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, regularization=None, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 不再添加 L1 正则项
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # 针对 early_stopping 的验证
        if regularization == 'early_stopping' and val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

# 测试函数（在 GPU 上运行）
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 定义不同规模卷积层配置
conv_filters_candidates = [
    [16, 32],    # 小型网络
    [32, 64],    # 中型网络
    [64, 128]    # 大型网络
]

# 目标函数（贝叶斯优化）--注意此处只采用 'none'、'dropout' 与 'early_stopping' 三种正则化方式
def objective(lr, num_epochs, batch_size, k_folds, conv_filters_index, momentum, dropout_rate):
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    conv_filters_index = int(conv_filters_index)
    conv_filters = conv_filters_candidates[conv_filters_index]
    regularization_types = ['none', 'dropout', 'early_stopping']
    fold_accuracies = {reg: [] for reg in regularization_types}

    torch.manual_seed(42)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f'Fold {fold+1}/{k_folds}')
        train_images_fold = train_images[train_idx]
        train_labels_fold = train_labels[train_idx]
        val_images_fold = train_images[val_idx]
        val_labels_fold = train_labels[val_idx]

        train_dataset_fold = torch.utils.data.TensorDataset(train_images_fold, train_labels_fold)
        val_dataset_fold = torch.utils.data.TensorDataset(val_images_fold, val_labels_fold)

        train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        val_loader_fold = torch.utils.data.DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

        for reg in regularization_types:
            print(f'Training with {reg} regularization')
            if reg == 'dropout':
                model = BasicCNN(conv_filters, dropout_rate=dropout_rate).to(device)
            else:
                model = BasicCNN(conv_filters).to(device)

            criterion = nn.CrossEntropyLoss()
            # 固定使用 SGD 优化器（无额外 weight_decay）
            optimizer_ = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

            train(model, train_loader_fold, val_loader_fold if reg == 'early_stopping' else None,
                  criterion, optimizer_, num_epochs, regularization=reg)
            val_accuracy = test(model, val_loader_fold)
            fold_accuracies[reg].append(val_accuracy)
            print(f'Fold {fold+1} {reg} Validation Accuracy: {val_accuracy:.2f}%')

    avg_val_accuracies = {reg: np.mean(acc) for reg, acc in fold_accuracies.items()}
    for reg, avg_acc in avg_val_accuracies.items():
        print(f'Average {reg} Validation Accuracy: {avg_acc:.2f}%')
    return avg_val_accuracies['none']

# 定义贝叶斯优化的超参数边界（去除了 l1_lambda 与 weight_decay）
pbounds = {
    'lr': (0.001, 0.1),
    'num_epochs': (5, 15),
    'batch_size': (16, 128),
    'k_folds': (3, 5),
    'conv_filters_index': (0, len(conv_filters_candidates) - 1),
    'momentum': (0.5, 0.99),
    'dropout_rate': (0.0, 0.5)
}

optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

# 执行贝叶斯优化
print("Starting Bayesian Optimization...")
optimizer.maximize(init_points=5, n_iter=10)

# 打印最佳超参数
print("\n=== Best Hyperparameters ===")
best_params = optimizer.max
print(f"Best Average Validation Accuracy (no regularization): {best_params['target']:.2f}%")
print(f"Best Parameters: {best_params['params']}")

# 使用最佳超参数训练最终模型并比较不同正则化方式
def train_final_model(params):
    print("\nTraining final model with best hyperparameters...")
    torch.manual_seed(42)

    lr = params['lr']
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])
    conv_filters_index = int(params['conv_filters_index'])
    momentum = params['momentum']
    dropout_rate = params['dropout_rate']
    conv_filters = conv_filters_candidates[conv_filters_index]

    print(f"Selected conv_filters configuration: {conv_filters}")
    print(f"Selected momentum: {momentum}")
    print(f"Selected dropout_rate: {dropout_rate}")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 这里为了简单起见，使用训练集作为验证集
    val_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    regularization_types = ['none', 'dropout', 'early_stopping']
    results = {}

    for reg in regularization_types:
        print(f"\nRunning final model with {reg} regularization")
        if reg == 'dropout':
            model = BasicCNN(conv_filters, dropout_rate=dropout_rate).to(device)
        else:
            model = BasicCNN(conv_filters).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer_ = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        train(model, train_loader, val_loader if reg == 'early_stopping' else None,
              criterion, optimizer_, num_epochs, regularization=reg)

        test_accuracy = test(model, test_loader)
        results[reg] = test_accuracy
        print(f'Final Test Accuracy with {reg}: {test_accuracy:.2f}%')

    # 打印最终结果表
    df = pd.DataFrame(list(results.items()), columns=['Regularization', 'Test Accuracy (%)'])
    print("\n=== Final Results Table ===")
    print(df)

# 使用最佳参数训练最终模型
train_final_model(best_params['params'])