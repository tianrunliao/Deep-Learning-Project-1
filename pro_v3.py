import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from scipy.ndimage import shift, rotate, zoom

# ---------------------------
# 1. 自定义加载MNIST数据集
# ---------------------------
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

# 数据路径（请根据实际情况修改路径）
train_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# 加载数据（均为numpy数组，形状：(N, 28, 28)）
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# ---------------------------
# 2. 数据增强
# ---------------------------
def augment_data(images):
    """
    对每个 MNIST 图像生成3种增强：平移、旋转和缩放，返回所有增强图像。
    """
    augmented_images = []
    for image in images:
        # 平移：随机移动-2到2个像素
        shift_x, shift_y = np.random.randint(-2, 3, size=2)
        shifted_image = shift(image, [shift_x, shift_y], mode='nearest')
        augmented_images.append(shifted_image)

        # 旋转：随机旋转-10到10度
        angle = np.random.uniform(-10, 10)
        rotated_image = rotate(image, angle, reshape=False, mode='nearest')
        augmented_images.append(rotated_image)

        # 缩放：随机缩放0.9到1.1倍
        scale = np.random.uniform(0.9, 1.1)
        scaled_image = zoom(image, scale, mode='nearest')
        # 确保图像保持28x28大小
        if scaled_image.shape != (28, 28):
            if scaled_image.shape[0] > 28:
                scaled_image = scaled_image[:28, :28]
            else:
                pad_h = 28 - scaled_image.shape[0]
                pad_w = 28 - scaled_image.shape[1]
                scaled_image = np.pad(scaled_image, ((0, pad_h), (0, pad_w)), mode='constant')
        augmented_images.append(scaled_image)
    return np.array(augmented_images)

# 对训练图像进行数据增强（每张原图生成3个增强版本）
augmented_train_images = augment_data(train_images)
# 增强标签：由于一张图生成3张增强图，所以对每个标签复制3次
augmented_train_labels = np.repeat(train_labels, 3, axis=0)

# 合并原始训练图像和增强图像
combined_train_images = np.concatenate((train_images, augmented_train_images), axis=0)
combined_train_labels = np.concatenate((train_labels, augmented_train_labels), axis=0)

# ---------------------------
# 3. 数据预处理：归一化并转换为 PyTorch 张量
# ---------------------------
# 归一化到 [0,1] 并转换为 float32 张量
combined_train_images = torch.tensor(combined_train_images, dtype=torch.float32) / 255.0
test_images = torch.tensor(test_images, dtype=torch.float32) / 255.0

# 标签转换：long 类型
combined_train_labels = torch.tensor(combined_train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 将图像展平为1D向量（28x28 -> 784）
combined_train_images = combined_train_images.view(-1, 28 * 28)
test_images = test_images.view(-1, 28 * 28)

# 将标签转换为 one-hot 编码（用于 MSE 损失）
def to_one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

combined_train_labels_one_hot = to_one_hot(combined_train_labels)
test_labels_one_hot = to_one_hot(test_labels)

# 创建测试集 DataLoader
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels_one_hot)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------------
# 4. 定义一个简单的 MLP 模型
# ---------------------------
class BasicMLP(nn.Module):
    def __init__(self):
        super(BasicMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),  # 输入层到隐藏层1：128 个神经元
            nn.Sigmoid(),             # Sigmoid 激活
            nn.Linear(128, 64),       # 隐藏层1到隐藏层2：64 个神经元
            nn.Sigmoid(),
            nn.Linear(64, 10)         # 隐藏层2到输出层：10 个类别
        )
    
    def forward(self, x):
        return self.layers(x)

# ---------------------------
# 5. 定义训练和测试函数
# ---------------------------
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

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels, 1)  # one-hot 转换回类别
            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# ---------------------------
# 6. 定义目标函数，供贝叶斯优化调用
# ---------------------------
def objective(lr, num_epochs, batch_size, k_folds):
    # 将超参数转为整型
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    
    # 设置随机种子确保结果可重复
    torch.manual_seed(42)
    
    # 采用 k 折交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    
    # 注意这里使用合并后的训练数据（含增强）
    for fold, (train_idx, val_idx) in enumerate(kf.split(combined_train_images)):
        print(f'Fold {fold+1}/{k_folds}')
        
        # 划分训练和验证数据
        train_images_fold = combined_train_images[train_idx]
        train_labels_fold = combined_train_labels_one_hot[train_idx]
        val_images_fold = combined_train_images[val_idx]
        val_labels_fold = combined_train_labels_one_hot[val_idx]
        
        # 构造 DataLoader
        train_dataset_fold = torch.utils.data.TensorDataset(train_images_fold, train_labels_fold)
        val_dataset_fold = torch.utils.data.TensorDataset(val_images_fold, val_labels_fold)
        train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        val_loader_fold = torch.utils.data.DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)
        
        # 初始化模型、损失和优化器（每个 fold 重新初始化模型）
        model = BasicMLP()
        criterion = nn.MSELoss()  # 使用 MSE 损失
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # 训练模型
        train(model, train_loader_fold, criterion, optimizer, num_epochs)
        
        # 在验证集上评估
        val_accuracy = test(model, val_loader_fold)
        fold_accuracies.append(val_accuracy)
        print(f'Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')
    
    # 返回平均验证准确率（贝叶斯优化时最大化该值）
    avg_val_accuracy = np.mean(fold_accuracies)
    print(f'Average Validation Accuracy: {avg_val_accuracy:.2f}%')
    return avg_val_accuracy

# ---------------------------
# 7. 设置贝叶斯优化搜索空间并开始调参
# ---------------------------
pbounds = {
    'lr': (0.001, 0.1),         # 学习率范围
    'num_epochs': (5, 15),        # 训练轮数范围
    'batch_size': (16, 128),      # 批量大小范围
    'k_folds': (3, 5)             # 交叉验证折数范围
}

bayes_optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

print("Starting Bayesian Optimization...")
bayes_optimizer.maximize(
    init_points=5,  # 初始化随机点数
    n_iter=10,      # 迭代次数
)

# 打印最佳超参数
print("\n=== Best Hyperparameters ===")
best_params = bayes_optimizer.max
print(f"Best Average Validation Accuracy: {best_params['target']:.2f}%")
print(f"Best Parameters: {best_params['params']}")

# ---------------------------
# 8. 使用最佳超参数训练最终模型，并在测试集上评估
# ---------------------------
def train_final_model(best_params):
    print("\nTraining final model with best hyperparameters...")
    torch.manual_seed(42)
    
    # 提取超参数
    lr = best_params['lr']
    num_epochs = int(best_params['num_epochs'])
    batch_size = int(best_params['batch_size'])
    
    # 创建训练数据加载器（使用合并后的训练数据）
    train_dataset = torch.utils.data.TensorDataset(combined_train_images, combined_train_labels_one_hot)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型、损失和优化器
    final_model = BasicMLP()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(final_model.parameters(), lr=lr)
    
    # 训练最终模型
    train(final_model, train_loader, criterion, optimizer, num_epochs)
    
    # 在测试集上评估
    print("\nTesting the final model on the test set...")
    test_accuracy = test(final_model, test_loader)
    print(f'Final Test Accuracy with best parameters: {test_accuracy:.2f}%')

train_final_model(best_params['params'])