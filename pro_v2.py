import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from scipy.ndimage import shift, rotate, zoom

# 设置随机种子以确保结果可重复
torch.manual_seed(321)

####################################
# 1. 数据加载与预处理
####################################

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

# 加载数据（得到numpy数组，形状为(N, 28, 28)）
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

####################################
# 2. 数据增强：生成平移、旋转和缩放后的图像
####################################

def augment_data(images):
    """
    对每张图像生成3种增强变换：平移、旋转和缩放，返回所有增强后的图像。
    """
    augmented_images = []
    for image in images:
        # 平移：随机移动1-2个像素（-2到2）
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

# 对训练集进行数据增强，得到增强后的图像数组（每张原图生成3张增强图，总数为原来的3倍）
augmented_train_images = augment_data(train_images)

# 同时，将原始训练标签复制相应次数（这里每张图生成3张增强图）
augmented_train_labels = np.repeat(train_labels, 3, axis=0)

# 将原始图像与增强图像合并，作为最终的训练数据
combined_train_images = np.concatenate((train_images, augmented_train_images), axis=0)
combined_train_labels = np.concatenate((train_labels, augmented_train_labels), axis=0)

####################################
# 3. 转换为PyTorch张量并预处理
####################################

# 将图像归一化到[0, 1]，并转换为浮点类型的张量
combined_train_images = torch.tensor(combined_train_images, dtype=torch.float32) / 255.0
test_images = torch.tensor(test_images, dtype=torch.float32) / 255.0

# 将标签转换为long类型张量
combined_train_labels = torch.tensor(combined_train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 将图像展平为1D向量（28x28 -> 784）
combined_train_images = combined_train_images.view(-1, 28 * 28)
test_images = test_images.view(-1, 28 * 28)

# 定义函数将标签转换为one-hot编码（用于MSE损失）
def to_one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

combined_train_labels_one_hot = to_one_hot(combined_train_labels)
test_labels_one_hot = to_one_hot(test_labels)

# 创建测试集的数据加载器
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels_one_hot)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

####################################
# 4. 定义一个基本的MLP模型（未做优化，固定隐藏层结构）
####################################

class BasicMLP(nn.Module):
    def __init__(self):
        super(BasicMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),  # 输入层到第一个隐藏层，128个神经元
            nn.Sigmoid(),             # 使用Sigmoid激活函数
            nn.Linear(128, 64),       # 第一个隐藏层到第二个隐藏层，64个神经元
            nn.Sigmoid(),
            nn.Linear(64, 10)         # 第二个隐藏层到输出层（10个类别）
        )
    
    def forward(self, x):
        return self.layers(x)

####################################
# 5. 定义训练和测试函数
####################################

def train(model, train_loader, criterion, optimizer, num_epochs=5):
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
            _, true_labels = torch.max(labels, 1)  # 从one-hot转换回类别
            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

####################################
# 6. 5折交叉验证
####################################

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_accuracies = []

print("Starting 5-fold cross-validation...")
# 注意：kf.split()的输入可以是任意数组，这里以combined_train_images为例
for fold, (train_idx, val_idx) in enumerate(kf.split(combined_train_images)):
    print(f'\nFold {fold+1}/{k}')

    # 划分训练和验证数据
    train_images_fold = combined_train_images[train_idx]
    train_labels_fold = combined_train_labels_one_hot[train_idx]
    val_images_fold = combined_train_images[val_idx]
    val_labels_fold = combined_train_labels_one_hot[val_idx]

    # 创建数据加载器
    train_dataset_fold = torch.utils.data.TensorDataset(train_images_fold, train_labels_fold)
    val_dataset_fold = torch.utils.data.TensorDataset(val_images_fold, val_labels_fold)
    train_loader_fold = torch.utils.data.DataLoader(train_dataset_fold, batch_size=64, shuffle=True)
    val_loader_fold = torch.utils.data.DataLoader(val_dataset_fold, batch_size=64, shuffle=False)

    # 初始化模型、损失函数和优化器（每次fold均重新初始化模型）
    model = BasicMLP()
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 简单SGD，无动量

    # 训练当前fold模型
    train(model, train_loader_fold, criterion, optimizer, num_epochs=5)
    # 在验证集上评估
    val_accuracy = test(model, val_loader_fold)
    fold_accuracies.append(val_accuracy)
    print(f'Fold {fold+1} Validation Accuracy: {val_accuracy:.2f}%')

# 计算交叉验证的平均准确率
avg_val_accuracy = np.mean(fold_accuracies)
print(f'\nAverage Validation Accuracy across {k} folds: {avg_val_accuracy:.2f}%')

####################################
# 7. 使用所有训练数据重新训练最终模型，并在测试集上评估
####################################

print("\nTraining final model on all training data...")
final_model = BasicMLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(final_model.parameters(), lr=0.01)

train_dataset = torch.utils.data.TensorDataset(combined_train_images, combined_train_labels_one_hot)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
train(final_model, train_loader, criterion, optimizer, num_epochs=5)

print("\nTesting the final model on the test set...")
test_accuracy = test(final_model, test_loader)