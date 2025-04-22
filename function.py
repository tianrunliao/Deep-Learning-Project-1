import torch
import torch.nn as nn
import gzip
import numpy as np
import struct
from scipy.ndimage import shift, rotate, zoom
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import pandas as pd

# --- GPU 相关辅助函数 ---

def setup_gpu():
    """设置并检查GPU可用性"""
    if not torch.cuda.is_available():
        print("警告: 未检测到可用的CUDA设备，将使用CPU运行")
        return False, 'cpu'
    
    # 获取可用的GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU设备")
    
    # 打印GPU信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"GPU {i}: {gpu_name}，内存: {total_memory:.2f} GB")
    
    # 设置设备为第一个GPU
    device = 'cuda:0'
    torch.cuda.set_device(0)
    print(f"使用设备: {device}")
    
    # 打印当前GPU内存使用情况
    print_gpu_memory_usage()
    
    return True, device

def print_gpu_memory_usage():
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(current_device) / 1024**3  # GB
        allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # GB
        free = total_memory - allocated
        
        print(f"GPU内存使用: 总计 {total_memory:.2f} GB, 已分配 {allocated:.2f} GB, 预留 {reserved:.2f} GB, 可用 {free:.2f} GB")

def clear_gpu_memory():
    """清理GPU缓存以释放内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU缓存")
        print_gpu_memory_usage()

# MLP 模型
class BasicMLP(nn.Module):
    def __init__(self, nHidden, dropout_rate=0.0, activation_fn='sigmoid'):
        super(BasicMLP, self).__init__()
        layers = []
        input_dim = 28 * 28
        
        # 获取激活函数
        activation = get_activation_function(activation_fn)
        
        for hidden_units in nHidden:
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_units
        
        layers.append(nn.Linear(input_dim, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# CNN 模型
class BasicCNN(nn.Module):
    def __init__(self, conv_filters, dropout_rate=0.0, activation_fn='relu', pool_type='max'):
        super(BasicCNN, self).__init__()
        
        # 获取激活函数
        activation = get_activation_function(activation_fn)
        
        # 选择池化类型
        if pool_type == 'max':
            pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:  # 'avg'
            pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels=1, out_channels=conv_filters[0], kernel_size=3, padding=1),
            activation,
            pooling,  # 输出: [N, conv_filters[0], 14, 14]
            
            # 第二层卷积
            nn.Conv2d(in_channels=conv_filters[0], out_channels=conv_filters[1], kernel_size=3, padding=1),
            activation,
            pooling   # 输出: [N, conv_filters[1], 7, 7]
        )
        
        # 展平后特征的维度
        self.flatten_dim = conv_filters[1] * 7 * 7
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            activation,
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(128, 10)  # 输出：10 类
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

def get_mlp_structure(structure_option='medium'):
    """
    获取 MLP 结构配置
    
    Args:
        structure_option: 'small', 'medium', 或 'large'
        
    Returns:
        包含隐藏层神经元数量的列表
    """
    structures = {
        'small': [128],                # 一个隐藏层，128个神经元
        'medium': [256, 128],          # 两个隐藏层
        'large': [512, 256, 128]       # 三个隐藏层
    }
    return structures.get(structure_option, [256, 128])  # 默认使用 medium 结构

def get_cnn_structure(structure_option='medium'):
    """
    获取 CNN 结构配置
    
    Args:
        structure_option: 'small', 'medium', 或 'large'
        
    Returns:
        包含卷积层滤波器数量的列表
    """
    structures = {
        'small': [16, 32],      # 小型网络
        'medium': [32, 64],     # 中型网络
        'large': [64, 128]      # 大型网络
    }
    return structures.get(structure_option, [32, 64])  # 修正为正确的数值

def create_model(model_type, structure_option='medium', dropout_rate=0.0, activation_fn='relu', pool_type='max'):
    """
    创建指定类型和结构的模型
    
    Args:
        model_type: 'mlp' 或 'cnn'
        structure_option: 'small', 'medium', 或 'large'
        dropout_rate: Dropout 比例
        activation_fn: 激活函数类型
        pool_type: 池化层类型 ('max' 或 'avg')，仅用于 CNN
        
    Returns:
        创建好的模型实例
    """
    if model_type.lower() == 'mlp':
        structure = get_mlp_structure(structure_option)
        return BasicMLP(structure, dropout_rate, activation_fn)
    elif model_type.lower() == 'cnn':
        structure = get_cnn_structure(structure_option)
        return BasicCNN(structure, dropout_rate, activation_fn, pool_type)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
def get_activation_function(activation_type='relu'):
    """
    获取指定类型的激活函数
    
    Args:
        activation_type: 激活函数类型名称
        
    Returns:
        对应的 PyTorch 激活函数模块
    """
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'elu': nn.ELU(),
    }
    return activations.get(activation_type.lower(), nn.ReLU())  # 默认返回 ReLU

def get_loss_function(loss_type='cross_entropy', reduction='mean', use_augmented=False):
    """
    获取指定类型的损失函数，针对增强数据可以使用加权损失
    """
    if use_augmented and loss_type.lower() == 'cross_entropy':
        # 对于增强数据使用标签平滑技术，减轻过拟合
        return nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
    
    losses = {
        'mse': nn.MSELoss(reduction=reduction),
        'cross_entropy': nn.CrossEntropyLoss(reduction=reduction),
        'bce': nn.BCELoss(reduction=reduction),
        'bce_with_logits': nn.BCEWithLogitsLoss(reduction=reduction),
        'l1': nn.L1Loss(reduction=reduction),
    }
    return losses.get(loss_type.lower(), nn.CrossEntropyLoss(reduction=reduction))

def get_optimizer(optimizer_type='sgd', model_parameters=None, lr=0.01, momentum=0.9, weight_decay=0.0, betas=(0.9, 0.999)):
    """
    获取指定类型的优化器
    
    Args:
        optimizer_type: 优化器类型名称
        model_parameters: 模型参数
        lr: 学习率
        momentum: SGD 动量参数
        weight_decay: L2 正则化强度
        betas: Adam系列优化器的 beta 参数
        
    Returns:
        配置好的 PyTorch 优化器
    """
    if model_parameters is None:
        raise ValueError("必须提供模型参数!")
        
    optimizers = {
        'sgd': lambda: torch.optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay),
        'adam': lambda: torch.optim.Adam(model_parameters, lr=lr, betas=betas, weight_decay=weight_decay),
        'adamw': lambda: torch.optim.AdamW(model_parameters, lr=lr, betas=betas, weight_decay=weight_decay),
        'rmsprop': lambda: torch.optim.RMSprop(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay),
    }
    
    optimizer_fn = optimizers.get(optimizer_type.lower(), optimizers['sgd'])
    return optimizer_fn()

def apply_l1_regularization(model, loss, l1_lambda=0.001):
    """
    应用 L1 正则化
    
    Args:
        model: 神经网络模型
        loss: 当前损失值
        l1_lambda: L1 正则化强度
        
    Returns:
        添加了 L1 正则项的损失值
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return loss + l1_lambda * l1_norm

def configure_regularization(config):
    """
    配置正则化策略
    
    Args:
        config: 包含正则化配置的字典
        
    Returns:
        配置好的正则化选项字典
    """
    regularization_config = {
        'methods': config.get('regularization_methods', ['none']),
        'dropout_rate': config.get('dropout_rate', 0.0),
        'l1_lambda': config.get('l1_lambda', 0.0001),
        'weight_decay': config.get('weight_decay', 0.0001),  # 用于 L2 正则化
        'early_stopping': config.get('early_stopping', False),
        'patience': config.get('patience', 5),
    }
    return regularization_config

def load_mnist_data(images_path, labels_path, is_train=True):
    """
    加载 MNIST 数据集
    
    Args:
        images_path: 图像数据文件路径
        labels_path: 标签数据文件路径
        is_train: 是否为训练集
        
    Returns:
        加载的图像和标签数据
    """
    # 加载图像
    with gzip.open(images_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    
    # 加载标签
    with gzip.open(labels_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return images, labels

def load_augmented_data(images_path, labels_path):
    """
    加载预处理的增强数据集（由 train_set_adjust.py 生成）
    
    Args:
        images_path: 增强后图像数据文件路径
        labels_path: 增强后标签数据文件路径
        
    Returns:
        加载的增强图像和标签数据
    """
    # 加载 npy.gz 格式的增强数据
    with gzip.open(images_path, 'rb') as f:
        images = np.load(f)
    
    with gzip.open(labels_path, 'rb') as f:
        labels = np.load(f)
    
    return images, labels

def apply_real_time_augmentation(dataset_type='mlp'):
    """
    配置实时数据增强管道
    
    Args:
        dataset_type: 'mlp' 或 'cnn' (影响是否保留通道维度)
        
    Returns:
        对应的 transforms 对象
    """
    if dataset_type == 'mlp':
        # MLP 需要展平为 1D
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # 展平为 1D
        ])
    else:  # CNN
        # CNN 保留 2D 结构
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ])
    
    return transform

def prepare_data_loaders(config):
    """根据配置准备数据加载器"""
    use_augmented = config.get('use_augmented_data', False)
    model_type = config.get('model_type', 'mlp').lower()
    batch_size = config.get('batch_size', 64)
    use_one_hot = config.get('use_one_hot', False)
    
    # 数据路径
    data_dir = config.get('data_dir', '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST')
    
    # 不管是否启用增强，都使用原始数据
    train_images_path = f"{data_dir}/train-images-idx3-ubyte.gz"
    train_labels_path = f"{data_dir}/train-labels-idx1-ubyte.gz"
    train_images, train_labels = load_mnist_data(train_images_path, train_labels_path)
    
    # 测试数据
    test_images_path = f"{data_dir}/t10k-images-idx3-ubyte.gz"
    test_labels_path = f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    test_images, test_labels = load_mnist_data(test_images_path, test_labels_path, is_train=False)
    
    # 转换为PyTorch张量
    train_images = torch.tensor(train_images, dtype=torch.float32) / 255.0
    test_images = torch.tensor(test_images, dtype=torch.float32) / 255.0
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # 根据模型类型调整数据形状
    if model_type == 'mlp':
        train_images = train_images.view(-1, 28 * 28)
        test_images = test_images.view(-1, 28 * 28)
    else:  # CNN
        train_images = train_images.unsqueeze(1)  # 添加通道维度 [N, 1, 28, 28]
        test_images = test_images.unsqueeze(1)
    
    # 如果启用增强，使用自定义数据集实现实时增强
    if use_augmented and model_type == 'cnn':  # 仅为CNN模型启用增强
        train_dataset = MNISTAugmentDataset(train_images, train_labels)
    else:
        # 使用常规数据集
        train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 添加一个新的自定义数据集类用于实时增强
class MNISTAugmentDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        # 仅使用非常轻微的增强
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=5,          # 最大旋转5度
                translate=(0.05, 0.05),  # 最大平移5%
                scale=(0.95, 1.05)       # 最大缩放5%
            ),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 50%概率应用增强
        if torch.rand(1).item() < 0.5:
            # 从[C,H,W]转为[H,W,C]再转回来
            image = self.transform(image.squeeze(0))  # 返回的是[C,H,W]形式的张量
        
        return image, label

def train_model(model, train_loader, val_loader=None, config=None, progress_callback=None):
    """
    训练神经网络模型
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器 (可选，用于早停)
        config: 训练配置
        progress_callback: 进度回调函数
        
    Returns:
        训练后的模型和训练历史记录
    """
    if config is None:
        config = {}
    
    # 提取训练参数
    device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    num_epochs = config.get('num_epochs', 10)
    loss_type = config.get('loss_type', 'cross_entropy')
    optimizer_type = config.get('optimizer_type', 'sgd')
    lr = config.get('learning_rate', 0.01)
    momentum = config.get('momentum', 0.9)
    weight_decay = config.get('weight_decay', 0.0)
    
    # 获取损失函数和优化器
    criterion = get_loss_function(loss_type)
    optimizer = get_optimizer(optimizer_type, model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [] if val_loader else None
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # 计算进度并每10%调用一次回调
        current_progress = int((epoch / num_epochs) * 100)
        if progress_callback and (current_progress % 10 == 0 or epoch == num_epochs - 1):
            progress_callback(epoch, num_epochs)
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)

            # --- 添加的修复逻辑 ---
            if loss_type == 'mse':
                # 检查标签是否已经是 one-hot (理论上不应该，但以防万一)
                if len(labels.shape) == 1:
                    try:
                        # 将类别索引转换为 one-hot 编码
                        # 需要知道类别数量，对于MNIST是10
                        num_classes = outputs.shape[1] # 从模型输出获取类别数更通用
                        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
                        labels_to_use = labels_one_hot
                    except Exception as e_onehot:
                        print(f"警告：将标签转换为 one-hot 时出错: {e_onehot}。标签形状: {labels.shape}")
                        labels_to_use = labels # 出错则使用原始标签，可能导致后续维度错误
                else:
                    labels_to_use = labels.float() # 确保数据类型是 float
            else:
                labels_to_use = labels # 对于非 MSE 损失，使用原始标签
            # --- 修复逻辑结束 ---
            
            # 计算损失 (使用转换后的标签)
            loss = criterion(outputs, labels_to_use)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 每 100 批次打印一次
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # 记录训练损失
        if len(train_loader) > 0:
            epoch_loss = running_loss / len(train_loader)
            history['train_loss'].append(epoch_loss)
        
        # 验证 (用于早停)
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    batch_loss = criterion(outputs, labels).item()
                    val_loss += batch_loss
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            print(f'Validation Loss: {val_loss:.4f}')
    
    return model, history

def test_model(model, test_loader, device=None):
    """
    测试神经网络模型
    
    Args:
        model: 神经网络模型
        test_loader: 测试数据加载器
        device: 运行设备
        
    Returns:
        测试准确率
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 如果标签是 one-hot 编码，转换回类索引
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                _, labels = torch.max(labels, 1)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return accuracy

def run_k_fold_validation(config, k=5):
    """
    执行 k 折交叉验证
    
    Args:
        config: 训练配置
        k: 折数
        
    Returns:
        平均验证准确率和折级别结果
    """
    model_type = config.get('model_type', 'mlp')
    structure_option = config.get('structure_option', 'medium')
    activation_fn = config.get('activation_fn', 'relu')
    pool_type = config.get('pool_type', 'max')
    dropout_rate = config.get('dropout_rate', 0.0)
    
    # 加载数据
    train_loader, _ = prepare_data_loaders(config)
    train_dataset = train_loader.dataset
    
    # 设置交叉验证
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    
    # 遍历每一折
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
        print(f'Fold {fold+1}/{k}')
        
        # 为当前折创建数据加载器
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)
        
        batch_size = config.get('batch_size', 64)
        train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # 创建新模型
        model = create_model(model_type, structure_option, dropout_rate, activation_fn, pool_type)
        
        # 训练模型
        fold_config = config.copy()
        model, _ = train_model(model, train_loader_fold, val_loader_fold, fold_config)
        
        # 验证模型
        accuracy = test_model(model, val_loader_fold, config.get('device'))
        fold_accuracies.append(accuracy)
        print(f'Fold {fold+1} Validation Accuracy: {accuracy:.2f}%')
    
    # 计算平均准确率
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f'Average Validation Accuracy: {avg_accuracy:.2f}%')
    
    return avg_accuracy, fold_accuracies

def objective_function(config):
    """
    优化目标函数 - 用于贝叶斯优化
    
    Args:
        config: 包含超参数的配置字典
        
    Returns:
        验证准确率 (用于优化)
    """
    # 提取超参数
    learning_rate = config.get('learning_rate', 0.01)
    num_epochs = int(config.get('num_epochs', 10))
    batch_size = int(config.get('batch_size', 64))
    k_folds = int(config.get('k_folds', 5))
    model_type = config.get('model_type', 'mlp')
    
    if model_type == 'mlp':
        structure_index = int(config.get('structure_index', 1))
        structure_options = ['small', 'medium', 'large']
        structure_option = structure_options[structure_index]
    else:  # CNN
        structure_index = int(config.get('structure_index', 1))
        structure_options = ['small', 'medium', 'large']
        structure_option = structure_options[structure_index]
    
    # 创建用于交叉验证的配置
    cv_config = {
        'model_type': model_type,
        'structure_option': structure_option,
        'activation_fn': config.get('activation_fn', 'relu'),
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer_type': config.get('optimizer_type', 'sgd'),
        'momentum': config.get('momentum', 0.9),
        'dropout_rate': config.get('dropout_rate', 0.0),
        'regularization_config': {
            'methods': config.get('regularization_methods', ['none']),
            'l1_lambda': config.get('l1_lambda', 0.0001),
            'weight_decay': config.get('weight_decay', 0.0001),
            'patience': config.get('patience', 5)
        },
        'device': config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'use_augmented_data': config.get('use_augmented_data', False),
        'data_dir': config.get('data_dir', '/path/to/data')
    }
    
    # 运行交叉验证
    avg_accuracy, _ = run_k_fold_validation(cv_config, k=k_folds)
    
    return avg_accuracy

def run_bayesian_optimization(config, init_points=5, n_iter=10):
    """
    运行贝叶斯优化寻找最佳超参数
    
    Args:
        config: 包含搜索边界和基本配置的字典
        init_points: 初始随机点数量
        n_iter: 迭代次数
        
    Returns:
        最佳超参数和对应的性能
    """
    # 提取搜索边界
    pbounds = config.get('pbounds', {
        'learning_rate': (0.001, 0.1),
        'num_epochs': (5, 15),
        'batch_size': (16, 128),
        'k_folds': (3, 5),
        'structure_index': (0, 2),  # 小型、中型或大型结构
        'momentum': (0.5, 0.99),
        'dropout_rate': (0.0, 0.5),
        'l1_lambda': (0.0001, 0.01),
        'weight_decay': (0.0001, 0.01)
    })
    
    # 创建优化器并设置基本配置
    base_config = {k: v for k, v in config.items() if k != 'pbounds'}
    
    def wrapped_objective(**params):
        """将参数合并到基本配置中，然后调用目标函数"""
        combined_config = base_config.copy()
        combined_config.update(params)
        return objective_function(combined_config)
    
    optimizer = BayesianOptimization(
        f=wrapped_objective,
        pbounds=pbounds,
        random_state=42
    )
    
    # 运行优化
    print("Starting Bayesian Optimization...")
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    # 返回结果
    best_params = optimizer.max
    print("\n=== Best Hyperparameters ===")
    print(f"Best Average Validation Accuracy: {best_params['target']:.2f}%")
    print(f"Best Parameters: {best_params['params']}")
    
    return best_params

def compare_regularization_methods(best_params, config):
    """
    使用最佳超参数比较不同的正则化方法
    
    Args:
        best_params: 贝叶斯优化找到的最佳超参数
        config: 基本配置
        
    Returns:
        不同正则化方法的性能比较表
    """
    # 结合最佳超参数和基本配置
    combined_config = config.copy()
    for key, value in best_params['params'].items():
        # 根据需要转换值的类型
        if key in ['num_epochs', 'batch_size', 'k_folds', 'structure_index']:
            combined_config[key] = int(value)
        else:
            combined_config[key] = value
    
    # 确定要比较的正则化方法
    regularization_methods = ['none', 'l1', 'l2', 'dropout', 'early_stopping']
    results = {}
    
    # 加载数据
    train_loader, test_loader = prepare_data_loaders(combined_config)
    
    # 一个接一个地比较每种正则化方法
    for method in regularization_methods:
        print(f"\nRunning with {method} regularization")
        
        # 创建一个独立的模型
        structure_options = ['small', 'medium', 'large']
        structure_option = structure_options[int(combined_config.get('structure_index', 1))]
        
        model = create_model(
            combined_config['model_type'],
            structure_option,
            dropout_rate=combined_config['dropout_rate'] if method == 'dropout' else 0.0,
            activation_fn=combined_config.get('activation_fn', 'relu')
        )
        
        # 配置特定的正则化方法
        reg_config = combined_config.copy()
        reg_config['regularization_config'] = {
            'methods': [method] if method != 'none' else [],
            'l1_lambda': combined_config.get('l1_lambda', 0.0001),
            'weight_decay': combined_config.get('weight_decay', 0.0001),
            'patience': combined_config.get('patience', 5)
        }
        
        # 训练模型
        model, _ = train_model(model, train_loader, train_loader, reg_config)  # 使用训练集作为验证集以简化
        
        # 测试模型
        accuracy = test_model(model, test_loader, combined_config.get('device'))
        results[method] = accuracy
    
    # 创建结果表
    df = pd.DataFrame(list(results.items()), columns=['Regularization', 'Test Accuracy (%)'])
    print("\n=== Final Results Table ===")
    print(df)
    
    return df

def main():
    """
    主函数
    """
    print("====== 神经网络训练配置 ======")
    
    # 步骤 1：选择模型类型
    model_types = ['MLP', 'CNN']
    model_type = get_user_choice("选择模型类型:", model_types).lower()
    
    # 步骤 2：选择是否使用增强数据
    use_augmentation = get_user_choice("使用增强训练数据?", ['是', '否'])
    use_augmented_data = (use_augmentation == '是')
    
    # 步骤 3：选择运行模式
    run_modes = ['使用推荐配置运行', '使用贝叶斯优化寻找最佳配置', '手动配置']
    run_mode = get_user_choice("选择运行模式:", run_modes)
    
    # 创建基本配置
    config = {
        'model_type': model_type,
        'use_augmented_data': use_augmented_data,
        'data_dir': '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 根据运行模式填充配置
    if run_mode == '使用推荐配置运行':
        # 填充推荐配置
        if model_type == 'mlp':
            config.update({
                'structure_option': 'medium',  # [256, 128]
                'activation_fn': 'sigmoid',
                'loss_type': 'cross_entropy',
                'optimizer_type': 'sgd',
                'learning_rate': 0.01,
                'momentum': 0.9,
                'num_epochs': 10,
                'batch_size': 64,
                'regularization_config': {
                    'methods': ['dropout', 'early_stopping'],
                    'dropout_rate': 0.3,
                    'patience': 5
                }
            })
        else:  # CNN
            config.update({
                'structure_option': 'medium',  # [32, 64]
                'activation_fn': 'relu',
                'pool_type': 'max',
                'loss_type': 'cross_entropy',
                'optimizer_type': 'adam',
                'learning_rate': 0.001,
                'num_epochs': 10,
                'batch_size': 64,
                'regularization_config': {
                    'methods': ['dropout', 'early_stopping'],
                    'dropout_rate': 0.3,
                    'patience': 5
                }
            })
        
        # 加载数据并训练模型
        train_loader, test_loader = prepare_data_loaders(config)
        model = create_model(
            config['model_type'],
            config['structure_option'],
            config.get('regularization_config', {}).get('dropout_rate', 0.0),
            config['activation_fn'],
            config.get('pool_type', 'max')
        )
        
        # 训练和测试
        model, _ = train_model(model, train_loader, train_loader, config)  # 使用训练集作为验证集
        accuracy = test_model(model, test_loader, config['device'])
        print(f"使用推荐配置的最终测试准确率: {accuracy:.2f}%")
    
    elif run_mode == '使用贝叶斯优化寻找最佳配置':
        # 配置贝叶斯优化边界
        if model_type == 'mlp':
            config['pbounds'] = {
                'learning_rate': (0.001, 0.1),
                'num_epochs': (5, 15),
                'batch_size': (16, 128),
                'k_folds': (3, 5),
                'structure_index': (0, 2),  # 小型、中型或大型结构
                'momentum': (0.5, 0.99),
                'dropout_rate': (0.0, 0.5),
                'l1_lambda': (0.0001, 0.01),
                'weight_decay': (0.0001, 0.01)
            }
        else:  # CNN
            config['pbounds'] = {
                'learning_rate': (0.001, 0.1),
                'num_epochs': (5, 15),
                'batch_size': (16, 128),
                'k_folds': (3, 5),
                'structure_index': (0, 2),  # 小型、中型或大型结构
                'momentum': (0.5, 0.99),
                'dropout_rate': (0.0, 0.5)
            }
        
        # 运行贝叶斯优化
        best_params = run_bayesian_optimization(config, init_points=5, n_iter=10)
        
        # 比较不同的正则化方法
        regularization_results = compare_regularization_methods(best_params, config)
        print("贝叶斯优化和正则化比较完成。")
    
    else:  # 手动配置
        # 结构选择
        structure_options = ['small', 'medium', 'large']
        config['structure_option'] = get_user_choice(
            f"选择 {model_type.upper()} 结构大小:", 
            structure_options
        )
        
        # 激活函数
        activation_options = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
        config['activation_fn'] = get_user_choice("选择激活函数:", activation_options)
        
        # 池化类型 (仅 CNN)
        if model_type == 'cnn':
            pool_options = ['max', 'avg']
            config['pool_type'] = get_user_choice("选择池化类型:", pool_options)
        
        # 损失函数
        loss_options = ['cross_entropy', 'mse']
        config['loss_type'] = get_user_choice("选择损失函数:", loss_options)
        
        # 优化器
        optimizer_options = ['sgd', 'adam', 'adamw', 'rmsprop']
        config['optimizer_type'] = get_user_choice("选择优化器:", optimizer_options)
        
        # 超参数
        config['learning_rate'] = get_user_float("输入学习率:", 0.01)
        config['num_epochs'] = get_user_int("输入训练轮数:", 10)
        config['batch_size'] = get_user_int("输入批量大小:", 64)
        
        if config['optimizer_type'] == 'sgd':
            config['momentum'] = get_user_float("输入动量:", 0.9)
        
        # 正则化
        reg_methods = []
        config['regularization_config'] = {'methods': reg_methods}
        
        # L1 正则化
        use_l1 = get_user_choice("使用 L1 正则化?", ['是', '否'])
        if use_l1 == '是':
            reg_methods.append('l1')
            config['regularization_config']['l1_lambda'] = get_user_float("输入 L1 lambda:", 0.0001)
        
        # L2 正则化
        use_l2 = get_user_choice("使用 L2 正则化?", ['是', '否'])
        if use_l2 == '是':
            reg_methods.append('l2')
            config['regularization_config']['weight_decay'] = get_user_float("输入 L2 weight decay:", 0.0001)
        
        # Dropout
        use_dropout = get_user_choice("使用 Dropout?", ['是', '否'])
        if use_dropout == '是':
            reg_methods.append('dropout')
            config['regularization_config']['dropout_rate'] = get_user_float("输入 Dropout 比率:", 0.3)
        
        # 早停
        use_early_stopping = get_user_choice("使用早停?", ['是', '否'])
        if use_early_stopping == '是':
            reg_methods.append('early_stopping')
            config['regularization_config']['patience'] = get_user_int("输入早停耐心值:", 5)
        
        # 加载数据并训练模型
        train_loader, test_loader = prepare_data_loaders(config)
        model = create_model(
            config['model_type'],
            config['structure_option'],
            config['regularization_config'].get('dropout_rate', 0.0) if 'dropout' in reg_methods else 0.0,
            config['activation_fn'],
            config.get('pool_type', 'max')
        )
        
        # 训练和测试
        model, _ = train_model(model, train_loader, train_loader, config)  # 使用训练集作为验证集
        accuracy = test_model(model, test_loader, config['device'])
        print(f"使用手动配置的最终测试准确率: {accuracy:.2f}%")
    
    print("程序执行完成。")

def get_user_choice(prompt, options):
    """
    获取用户输入的选项
    
    Args:
        prompt: 提示信息
        options: 选项列表
        
    Returns:
        用户选择的选项
    """
    print(prompt)
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    
    while True:
        try:
            choice = int(input("请输入选项编号: "))
            if 1 <= choice <= len(options):
                return options[choice-1]
            else:
                print("无效的选项编号。")
        except ValueError:
            print("请输入数字。")

def get_user_float(prompt, default=None):
    """
    获取用户输入的浮点数
    
    Args:
        prompt: 提示信息
        default: 默认值
        
    Returns:
        浮点数输入
    """
    while True:
        try:
            val_str = input(f"{prompt} (默认: {default}): ")
            if not val_str and default is not None:
                return default
            return float(val_str)
        except ValueError:
            print("请输入有效的数字。")

def get_user_int(prompt, default=None):
    """
    获取用户输入的整数
    
    Args:
        prompt: 提示信息
        default: 默认值
        
    Returns:
        整数输入
    """
    while True:
        try:
            val_str = input(f"{prompt} (默认: {default}): ")
            if not val_str and default is not None:
                return default
            return int(val_str)
        except ValueError:
            print("请输入有效的数字。")

