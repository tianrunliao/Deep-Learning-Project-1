import numpy as np
import gzip
import struct
from scipy.ndimage import shift, rotate, zoom
from sklearn.model_selection import KFold
import pandas as pd
import os # 添加 os 导入，用于 get_user_input_path

# --- NumPy 基础组件 ---

def sigmoid(x):
    """Sigmoid 激活函数"""
    # 防止 overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid 激活函数的导数"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 激活函数的导数"""
    return (x > 0).astype(float)

def tanh(x):
    """Tanh 激活函数"""
    return np.tanh(x)

def tanh_derivative(x):
    """Tanh 激活函数的导数"""
    return 1 - np.tanh(x)**2

def leaky_relu(x, alpha=0.1):
    """Leaky ReLU 激活函数"""
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivative(x, alpha=0.1):
    """Leaky ReLU 激活函数的导数"""
    return np.where(x > 0, 1, alpha)

def softmax(x):
    """Softmax 函数，处理数值稳定性"""
    # axis=1 表示对每一行（每个样本）进行 softmax
    # keepdims=True 保持维度以便广播
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred_softmax, y_true_indices):
    """
    计算交叉熵损失

    Args:
        y_pred_softmax: shape (N, C)，经过 softmax 的预测概率
        y_true_indices: shape (N,)，真实的类别索引

    Returns:
        平均交叉熵损失
    """
    n_samples = y_true_indices.shape[0]
    # 获取对应正确类别的预测概率
    correct_logprobs = -np.log(y_pred_softmax[np.arange(n_samples), y_true_indices] + 1e-9) # 加个小数防止log(0)
    loss = np.sum(correct_logprobs) / n_samples
    return loss

def cross_entropy_loss_derivative(y_pred_softmax, y_true_indices):
    """
    计算交叉熵损失相对于 softmax 输入（logits）的梯度

    Args:
        y_pred_softmax: shape (N, C)，经过 softmax 的预测概率
        y_true_indices: shape (N,)，真实的类别索引

    Returns:
        梯度 dLoss/dLogits，shape (N, C)
    """
    n_samples = y_true_indices.shape[0]
    # 创建 one-hot 编码的真实标签
    y_true_one_hot = np.zeros_like(y_pred_softmax)
    y_true_one_hot[np.arange(n_samples), y_true_indices] = 1

    # 梯度是 (y_pred_softmax - y_true_one_hot) / N
    grad = (y_pred_softmax - y_true_one_hot) / n_samples
    return grad

def mse_loss(y_pred, y_true_one_hot):
    """计算均方误差损失"""
    n_samples = y_true_one_hot.shape[0]
    loss = np.sum((y_pred - y_true_one_hot)**2) / (2 * n_samples)
    return loss

def mse_loss_derivative(y_pred, y_true_one_hot):
    """计算均方误差损失的梯度"""
    n_samples = y_true_one_hot.shape[0]
    grad = (y_pred - y_true_one_hot) / n_samples
    return grad

def one_hot_encode(labels, num_classes):
    """将类别索引转换为 one-hot 编码"""
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot


# --- NumPy 实现的 MLP 层 ---
class LinearLayer:
    def __init__(self, input_dim, output_dim):
        # He 初始化 / Xavier 初始化（简化版）
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        # self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim) # He
        self.biases = np.zeros((1, output_dim))
        self.input = None
        self.grad_weights = None
        self.grad_biases = None
        self.grad_input = None

    def forward(self, input_data):
        """前向传播"""
        self.input = input_data
        # (N, input_dim) @ (input_dim, output_dim) + (1, output_dim) -> (N, output_dim)
        output = np.dot(self.input, self.weights) + self.biases
        return output

    def backward(self, grad_output):
        """
        反向传播

        Args:
            grad_output: 输出层的梯度 (dLoss / dOutput)，shape (N, output_dim)

        Returns:
            输入层的梯度 (dLoss / dInput)，shape (N, input_dim)
        """
        # 计算权重的梯度: dLoss/dW = dLoss/dOutput * dOutput/dW = grad_output^T * input
        # (input_dim, N) @ (N, output_dim) -> (input_dim, output_dim)
        self.grad_weights = np.dot(self.input.T, grad_output)

        # 计算偏置的梯度: dLoss/dB = dLoss/dOutput * dOutput/dB = sum(grad_output, axis=0)
        # (N, output_dim) -> (1, output_dim)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)

        # 计算输入的梯度: dLoss/dInput = dLoss/dOutput * dOutput/dInput = grad_output @ W^T
        # (N, output_dim) @ (output_dim, input_dim) -> (N, input_dim)
        self.grad_input = np.dot(grad_output, self.weights.T)

        return self.grad_input

    def update(self, learning_rate):
        """使用 SGD 更新权重和偏置"""
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

class ActivationLayer:
    def __init__(self, activation_fn_name):
        self.activation_fn_name = activation_fn_name.lower()
        self.input = None
        self.grad_input = None

        activations = {
            'relu': (relu, relu_derivative),
            'sigmoid': (sigmoid, sigmoid_derivative),
            'tanh': (tanh, tanh_derivative),
            'leaky_relu': (leaky_relu, leaky_relu_derivative)
            # 可以添加更多
        }
        if self.activation_fn_name not in activations:
            raise ValueError(f"不支持的激活函数: {activation_fn_name}")
        self.activation_fn, self.activation_derivative = activations[self.activation_fn_name]

    def forward(self, input_data):
        """前向传播"""
        self.input = input_data
        output = self.activation_fn(self.input)
        return output

    def backward(self, grad_output):
        """
        反向传播

        Args:
            grad_output: 输出层的梯度 (dLoss / dOutput)，shape (N, layer_dim)

        Returns:
            输入层的梯度 (dLoss / dInput)，shape (N, layer_dim)
        """
        # 梯度: dLoss/dInput = dLoss/dOutput * dOutput/dInput
        # dOutput/dInput 是激活函数的导数
        self.grad_input = grad_output * self.activation_derivative(self.input)
        return self.grad_input

    def update(self, learning_rate):
        # 激活层没有参数需要更新
        pass


class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.is_training = True # 训练时启用 dropout，测试时禁用

    def forward(self, input_data):
        if not self.is_training or self.dropout_rate == 0:
            return input_data

        # 生成 dropout mask
        self.mask = (np.random.rand(*input_data.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
        # 应用 mask
        output = input_data * self.mask
        return output

    def backward(self, grad_output):
        if not self.is_training or self.dropout_rate == 0:
            return grad_output
        # 将梯度乘以相同的 mask
        grad_input = grad_output * self.mask
        return grad_input

    def update(self, learning_rate):
        # Dropout 层没有参数需要更新
        pass

    def set_training_mode(self, is_training):
        self.is_training = is_training


# --- NumPy 实现的 CNN 层 ---

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He 初始化 (简化)
        limit = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit
        self.biases = np.zeros((out_channels, 1)) # 每个输出通道一个偏置

        self.input = None
        self.input_padded = None # 存储填充后的输入
        self.grad_weights = None
        self.grad_biases = None
        self.grad_input = None

    def forward(self, input_data):
        """前向传播 (N, C_in, H_in, W_in) -> (N, C_out, H_out, W_out)"""
        self.input = input_data
        N, C_in, H_in, W_in = input_data.shape
        assert C_in == self.in_channels, "输入通道数与层定义不符"

        # 计算输出尺寸
        H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 添加 Padding
        if self.padding > 0:
            self.input_padded = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        else:
            self.input_padded = input_data

        # 初始化输出
        output = np.zeros((N, self.out_channels, H_out, W_out))

        # 卷积计算 (使用循环，效率较低)
        for n in range(N): # 遍历样本
            for c_out in range(self.out_channels): # 遍历输出通道
                for h in range(H_out): # 遍历输出高度
                    for w in range(W_out): # 遍历输出宽度
                        # 定位输入区域
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        # 提取输入区域 (所有输入通道)
                        input_slice = self.input_padded[n, :, h_start:h_end, w_start:w_end]

                        # 执行卷积: (C_in, K, K) * (C_in, K, K) -> 求和
                        conv_sum = np.sum(input_slice * self.weights[c_out, :, :, :])

                        # 添加偏置
                        output[n, c_out, h, w] = conv_sum + self.biases[c_out]

        return output

    def backward(self, grad_output):
        """
        反向传播 (N, C_out, H_out, W_out) -> (N, C_in, H_in, W_in)
        计算 dLoss/dInput, dLoss/dWeights, dLoss/dBiases
        """
        N, C_out, H_out, W_out = grad_output.shape
        _, _, H_in, W_in = self.input.shape # 原始输入尺寸

        # 初始化梯度
        self.grad_input = np.zeros_like(self.input_padded) # 梯度需要对应 padded 输入
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

        # 计算梯度 (使用循环，效率较低)
        for n in range(N): # 遍历样本
            for c_out in range(self.out_channels): # 遍历输出通道
                for h in range(H_out): # 遍历输出高度
                    for w in range(W_out): # 遍历输出宽度
                        # 获取当前输出位置的梯度
                        grad = grad_output[n, c_out, h, w]

                        # 定位输入区域
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        # 提取输入区域
                        input_slice = self.input_padded[n, :, h_start:h_end, w_start:w_end]

                        # --- 计算梯度 ---
                        # 1. dLoss/dInput: 将梯度乘权重，累加到对应输入位置
                        self.grad_input[n, :, h_start:h_end, w_start:w_end] += self.weights[c_out, :, :, :] * grad

                        # 2. dLoss/dWeights: 将梯度乘输入切片，累加到对应权重
                        self.grad_weights[c_out, :, :, :] += input_slice * grad

                        # 3. dLoss/dBiases: 将梯度累加到对应偏置
                        self.grad_biases[c_out] += grad

        # 如果有 padding，需要移除 padding 部分的梯度
        if self.padding > 0:
            self.grad_input = self.grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # 确保 grad_input 形状与原始输入一致
        assert self.grad_input.shape == self.input.shape

        return self.grad_input

    def update(self, learning_rate):
        """使用 SGD 更新权重和偏置"""
        self.weights -= learning_rate * self.grad_weights
        # grad_biases 是 (C_out, 1)，需要 reshape 或广播匹配 biases (C_out, 1)
        self.biases -= learning_rate * self.grad_biases # NumPy 应该能自动广播


class PoolingLayer:
    def __init__(self, pool_size, stride=None, pool_type='max'):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size # 默认 stride 等于 pool_size
        self.pool_type = pool_type.lower()
        if self.pool_type != 'max':
            raise NotImplementedError("当前仅支持 Max Pooling")

        self.input = None
        self.output_shape = None
        self.max_indices = None # 存储最大值索引用于反向传播
        self.grad_input = None

    def forward(self, input_data):
        """前向传播 (N, C, H_in, W_in) -> (N, C, H_out, W_out)"""
        self.input = input_data
        N, C, H_in, W_in = input_data.shape

        # 计算输出尺寸
        H_out = (H_in - self.pool_size) // self.stride + 1
        W_out = (W_in - self.pool_size) // self.stride + 1
        self.output_shape = (N, C, H_out, W_out)

        # 初始化输出和最大值索引存储
        output = np.zeros(self.output_shape)
        self.max_indices = np.zeros(self.output_shape, dtype=int) # 存储展平后的索引

        # Max Pooling 计算 (使用循环)
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        # 定位输入区域
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        # 提取输入区域
                        input_slice = self.input[n, c, h_start:h_end, w_start:w_end]

                        # 找到最大值及其在切片内的索引
                        max_val = np.max(input_slice)
                        max_idx_flat = np.argmax(input_slice) # 展平后的索引
                        max_idx_row, max_idx_col = np.unravel_index(max_idx_flat, (self.pool_size, self.pool_size))

                        # 存储输出和最大值在 *原始输入* 中的绝对索引 (需要转换)
                        output[n, c, h, w] = max_val
                        # 计算在原始 (H_in, W_in) 中的索引，然后展平成单个整数存储
                        abs_row = h_start + max_idx_row
                        abs_col = w_start + max_idx_col
                        self.max_indices[n, c, h, w] = abs_row * W_in + abs_col

        return output

    def backward(self, grad_output):
        """反向传播 (N, C, H_out, W_out) -> (N, C, H_in, W_in)"""
        N, C, H_out, W_out = grad_output.shape
        _, _, H_in, W_in = self.input.shape

        # 初始化输入梯度
        self.grad_input = np.zeros_like(self.input)

        # 将梯度传递回最大值所在的位置
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        # 获取当前输出位置的梯度
                        grad = grad_output[n, c, h, w]
                        # 获取最大值在原始输入中的展平索引
                        flat_idx = self.max_indices[n, c, h, w]
                        # 将展平索引转换回 (row, col)
                        idx_row, idx_col = np.unravel_index(flat_idx, (H_in, W_in))
                        # 将梯度累加到对应位置 (如果一个位置是多个池化窗口的最大值，梯度会累加)
                        self.grad_input[n, c, idx_row, idx_col] += grad

        return self.grad_input

    def update(self, learning_rate):
        pass # 池化层无参数

class FlattenLayer:
    def __init__(self):
        self.original_shape = None

    def forward(self, input_data):
        """前向传播 (N, C, H, W) -> (N, C*H*W)"""
        self.original_shape = input_data.shape
        N = self.original_shape[0]
        # 将除 N 以外的所有维度展平
        output = input_data.reshape(N, -1)
        return output

    def backward(self, grad_output):
        """反向传播 (N, C*H*W) -> (N, C, H, W)"""
        # 将梯度 reshape回原始输入的形状
        grad_input = grad_output.reshape(self.original_shape)
        return grad_input

    def update(self, learning_rate):
        pass # 展平层无参数


# --- NumPy 实现的 MLP 模型 ---
class BasicMLP_NumPy:
    def __init__(self, nHidden, dropout_rate=0.0, activation_fn='sigmoid'):
        self.layers = []
        input_dim = 28 * 28

        for i, hidden_units in enumerate(nHidden):
            # 添加线性层
            self.layers.append(LinearLayer(input_dim, hidden_units))
            # 添加激活层
            self.layers.append(ActivationLayer(activation_fn))
            # 添加 Dropout 层
            if dropout_rate > 0:
                self.layers.append(DropoutLayer(dropout_rate))
            input_dim = hidden_units

        # 添加最后的输出线性层 (没有激活函数，将在损失函数前应用 softmax)
        self.layers.append(LinearLayer(input_dim, 10))

    def forward(self, x):
        """模型的前向传播"""
        for layer in self.layers:
            x = layer.forward(x)
        return x # 返回 logits

    def backward(self, grad_output):
        """模型的反向传播"""
        # 从最后一层开始反向传播梯度
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output # 返回相对于模型输入的梯度 (通常不需要)

    def update(self, learning_rate):
        """更新模型中所有层的参数"""
        for layer in self.layers:
            layer.update(learning_rate)

    def set_training_mode(self, is_training):
        """设置模型及其层的训练/评估模式"""
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.set_training_mode(is_training)


# --- NumPy 实现的 CNN 模型 ---
class BasicCNN_NumPy:
    def __init__(self, conv_filters=(16, 32), kernel_size=3, pool_size=2,
                 fc_hidden_units=(64,), dropout_rate=0.0, activation_fn='relu'):
        self.layers = []
        in_channels = 1 # MNIST 输入通道为 1
        input_h, input_w = 28, 28 # MNIST 输入尺寸

        # 卷积和池化层
        for i, out_channels in enumerate(conv_filters):
            # 卷积层 (使用 valid padding)
            conv = ConvLayer(in_channels, out_channels, kernel_size, stride=1, padding=0) # 'valid'
            self.layers.append(conv)
            # 计算卷积后的尺寸
            input_h = (input_h - kernel_size) + 1
            input_w = (input_w - kernel_size) + 1

            # 激活层
            self.layers.append(ActivationLayer(activation_fn))

            # 池化层
            pool = PoolingLayer(pool_size)
            self.layers.append(pool)
            # 计算池化后的尺寸
            input_h = (input_h - pool_size) // pool_size + 1
            input_w = (input_w - pool_size) // pool_size + 1

            in_channels = out_channels # 更新下一层的输入通道

        # 展平层
        self.layers.append(FlattenLayer())
        flattened_dim = in_channels * input_h * input_w # 计算展平后的维度

        # 全连接层
        input_dim = flattened_dim
        for i, hidden_units in enumerate(fc_hidden_units):
            self.layers.append(LinearLayer(input_dim, hidden_units))
            self.layers.append(ActivationLayer(activation_fn))
            if dropout_rate > 0:
                self.layers.append(DropoutLayer(dropout_rate))
            input_dim = hidden_units

        # 输出层
        self.layers.append(LinearLayer(input_dim, 10))

    def forward(self, x):
        """模型的前向传播 (N, 1, 28, 28) -> (N, 10)"""
        # 确保输入是 4D
        if x.ndim == 2: # 如果输入是展平的 (e.g., from MLP test)
            N = x.shape[0]
            x = x.reshape(N, 1, 28, 28) # 假设是 MNIST
        elif x.ndim != 4:
             raise ValueError(f"CNN 输入应为 4D (N, C, H, W)，但得到 {x.ndim}D")

        for layer in self.layers:
            x = layer.forward(x)
        return x # 返回 logits

    def backward(self, grad_output):
        """模型的反向传播"""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def update(self, learning_rate):
        """更新模型中所有层的参数"""
        for layer in self.layers:
            layer.update(learning_rate)

    def set_training_mode(self, is_training):
        """设置模型及其层的训练/评估模式"""
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.set_training_mode(is_training)


# --- 模型和结构选择 ---

def get_mlp_structure(structure_option='medium'):
    structures = {'small': [128], 'medium': [256, 128], 'large': [512, 256, 128]}
    return structures.get(structure_option, [256, 128])

def get_cnn_structure(structure_option='medium'):
    # 返回卷积层的滤波器数量和全连接层的隐藏单元数量
    # (conv_filters, fc_hidden_units)
    structures = {
        'small': ([16], [32]),            # 1 Conv, 1 FC
        'medium': ([16, 32], [64]),       # 2 Conv, 1 FC
        'large': ([32, 64, 64], [128, 64]) # 3 Conv, 2 FC
    }
    return structures.get(structure_option, ([16, 32], [64]))

def create_model(model_type, structure_option='medium', dropout_rate=0.0, activation_fn='relu'):
    """创建指定类型和结构的模型"""
    model_type = model_type.lower()
    if model_type == 'mlp':
        structure = get_mlp_structure(structure_option)
        return BasicMLP_NumPy(structure, dropout_rate, activation_fn)
    elif model_type == 'cnn':
        conv_filters, fc_hidden_units = get_cnn_structure(structure_option)
        # 可以添加更多参数给 CNN，如 kernel_size, pool_size
        return BasicCNN_NumPy(conv_filters=conv_filters,
                              fc_hidden_units=fc_hidden_units,
                              dropout_rate=dropout_rate,
                              activation_fn=activation_fn)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# --- 损失函数和优化器 ---

def get_loss_function(loss_type='cross_entropy'):
    if loss_type.lower() == 'cross_entropy':
        return cross_entropy_loss, cross_entropy_loss_derivative
    elif loss_type.lower() == 'mse':
        return mse_loss, mse_loss_derivative
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")

class SGD_Optimizer:
    """简单的 SGD 优化器 (NumPy 版本)，支持 MLP 和 CNN"""
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
        self._initialize_velocities()

    def _initialize_velocities(self):
        """初始化动量速度字典"""
        self.velocities = {}
        if self.momentum > 0:
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                    # 适用于 LinearLayer 和 ConvLayer
                    self.velocities[f'w_{i}'] = np.zeros_like(layer.weights)
                    self.velocities[f'b_{i}'] = np.zeros_like(layer.biases)

    def step(self):
        """执行一步优化，更新模型参数"""
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weights') and hasattr(layer, 'biases') and \
               hasattr(layer, 'grad_weights') and hasattr(layer, 'grad_biases') and \
               layer.grad_weights is not None and layer.grad_biases is not None: # 确保梯度已计算

                if self.momentum > 0:
                    # 更新速度
                    self.velocities[f'w_{i}'] = self.momentum * self.velocities[f'w_{i}'] - self.learning_rate * layer.grad_weights
                    self.velocities[f'b_{i}'] = self.momentum * self.velocities[f'b_{i}'] - self.learning_rate * layer.grad_biases
                    # 使用速度更新参数
                    layer.weights += self.velocities[f'w_{i}']
                    layer.biases += self.velocities[f'b_{i}']
                else:
                    # 无动量的 SGD 更新
                    layer.weights -= self.learning_rate * layer.grad_weights
                    layer.biases -= self.learning_rate * layer.grad_biases
            elif isinstance(layer, (ActivationLayer, DropoutLayer, PoolingLayer, FlattenLayer)):
                 pass # 这些层没有可训练参数
            # else:
                 # print(f"警告: 层 {i} ({type(layer)}) 没有更新，可能缺少参数或梯度。")


def get_optimizer(optimizer_type='sgd', model=None, lr=0.01, momentum=0.9, weight_decay=0.0):
    if model is None: raise ValueError("必须提供模型实例!")
    if weight_decay > 0: print("警告: NumPy 版本的 SGD 优化器暂未实现 weight_decay")
    optimizer_type = optimizer_type.lower()
    if optimizer_type == 'sgd':
        return SGD_Optimizer(model, lr, momentum)
    elif optimizer_type in ['adam', 'adamw', 'rmsprop']:
        raise NotImplementedError(f"NumPy 版本的 {optimizer_type} 优化器尚未实现")
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

def configure_regularization(config):
    methods = config.get('regularization_methods', [])
    return {
        'methods': methods,
        'dropout_rate': config.get('dropout_rate', 0.0) if 'dropout' in methods else 0.0,
        'early_stopping': 'early_stopping' in methods,
        'patience': config.get('patience', 5),
    }

# --- 数据加载与处理 ---

def load_mnist_data(images_path, labels_path, model_type='mlp'):
    """加载 MNIST 数据集 (返回 NumPy 数组)"""
    with gzip.open(images_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        if model_type == 'cnn':
            # CNN 需要 (N, C, H, W) 格式， MNIST 是灰度图，C=1
            images = images.reshape(num_images, 1, rows, cols)
        else: # MLP 需要展平
            images = images.reshape(num_images, rows * cols)
        images = images.astype(np.float32) / 255.0 # 归一化

    with gzip.open(labels_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels

def prepare_data_loaders(config):
    """根据配置准备数据"""
    model_type = config.get('model_type', 'mlp').lower()
    batch_size = config.get('batch_size', 64)
    loss_type = config.get('loss_type', 'cross_entropy').lower()
    use_one_hot = loss_type == 'mse'
    data_dir = config.get('data_dir', './mnist_data') # 默认当前目录下的 mnist_data

    # 加载数据，根据模型类型确定形状
    try:
        train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
        train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
        train_images, train_labels = load_mnist_data(train_images_path, train_labels_path, model_type)

        test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
        test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
        test_images, test_labels = load_mnist_data(test_images_path, test_labels_path, model_type)
    except FileNotFoundError as e:
         print(f"错误：找不到 MNIST 数据文件。请确保路径 '{data_dir}' 正确并包含以下文件：")
         print("- train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz")
         print("- t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz")
         raise e

    # 如果使用 MSE，标签转换为 one-hot
    if use_one_hot:
        num_classes = 10
        train_labels = one_hot_encode(train_labels, num_classes)
        test_labels = one_hot_encode(test_labels, num_classes)

    # 准备加载器信息字典
    def create_loader_info(images, labels, shuffle):
        num_samples = images.shape[0]
        return {
            'images': images,
            'labels': labels,
            'batch_size': batch_size,
            'num_samples': num_samples,
            'num_batches': int(np.ceil(num_samples / batch_size)),
            'shuffle': shuffle,
            'use_one_hot': use_one_hot,
            'num_classes': 10 # MNIST specific
        }

    train_loader_info = create_loader_info(train_images, train_labels, True)
    test_loader_info = create_loader_info(test_images, test_labels, False)

    return train_loader_info, test_loader_info

def get_batch(loader_info, batch_index, indices=None):
    """从数据信息中获取一个批次"""
    start = batch_index * loader_info['batch_size']
    end = min(start + loader_info['batch_size'], loader_info['num_samples']) # 处理最后一个批次

    if indices is not None:
        batch_indices = indices[start:end]
        images = loader_info['images'][batch_indices]
        labels = loader_info['labels'][batch_indices]
    else:
        images = loader_info['images'][start:end]
        labels = loader_info['labels'][start:end]

    return images, labels


# --- 训练与测试 (NumPy 版本) ---

def train_model(model, train_loader_info, val_loader_info=None, config=None, progress_callback=None):
    """训练 NumPy 模型 (MLP 或 CNN)"""
    if config is None: config = {}

    num_epochs = config.get('num_epochs', 10)
    loss_type = config.get('loss_type', 'cross_entropy')
    optimizer_type = config.get('optimizer_type', 'sgd')
    lr = config.get('learning_rate', 0.01)
    momentum = config.get('momentum', 0.0) # 默认 0
    regularization_config = configure_regularization(config)

    loss_fn, loss_derivative_fn = get_loss_function(loss_type)
    optimizer = get_optimizer(optimizer_type, model, lr, momentum)

    history = {'train_loss': [], 'val_loss': [] if val_loader_info else None}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.set_training_mode(True)
        running_loss = 0.0

        indices = np.arange(train_loader_info['num_samples'])
        if train_loader_info['shuffle']:
            np.random.shuffle(indices)

        if progress_callback: progress_callback(epoch, num_epochs)

        for i in range(train_loader_info['num_batches']):
            batch_images, batch_labels = get_batch(train_loader_info, i, indices)

            logits = model.forward(batch_images)

            if loss_type == 'cross_entropy':
                probs = softmax(logits)
                loss = loss_fn(probs, batch_labels) # labels 是索引
                grad_loss = loss_derivative_fn(probs, batch_labels)
            elif loss_type == 'mse':
                loss = loss_fn(logits, batch_labels) # labels 是 one-hot
                grad_loss = loss_derivative_fn(logits, batch_labels)
            else: raise ValueError("不支持的损失类型")

            running_loss += loss

            model.backward(grad_loss)
            optimizer.step()

        epoch_loss = running_loss / train_loader_info['num_batches']
        history['train_loss'].append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

        # --- 验证 ---
        if val_loader_info:
            model.set_training_mode(False)
            val_loss = 0.0
            for i in range(val_loader_info['num_batches']):
                val_images, val_labels = get_batch(val_loader_info, i)
                val_logits = model.forward(val_images)

                if loss_type == 'cross_entropy':
                    val_probs = softmax(val_logits)
                    batch_val_loss = loss_fn(val_probs, val_labels)
                elif loss_type == 'mse':
                    batch_val_loss = loss_fn(val_logits, val_labels)
                val_loss += batch_val_loss

            epoch_val_loss = val_loss / val_loader_info['num_batches']
            history['val_loss'].append(epoch_val_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_val_loss:.4f}')

            # 早停逻辑
            if regularization_config['early_stopping']:
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= regularization_config['patience']:
                        print(f"早停触发!")
                        break

    model.set_training_mode(False)
    return model, history


def test_model(model, test_loader_info, loss_type='cross_entropy'):
    """测试 NumPy 模型 (MLP 或 CNN)"""
    model.set_training_mode(False)
    correct = 0
    total = 0

    for i in range(test_loader_info['num_batches']):
        images, labels_true = get_batch(test_loader_info, i) # 获取原始标签 (索引或 one-hot)

        logits = model.forward(images)
        predicted_indices = np.argmax(logits, axis=1)

        if test_loader_info['use_one_hot']: # 如果标签是 one-hot
            true_indices = np.argmax(labels_true, axis=1)
        else: # 标签是索引
            true_indices = labels_true

        total += true_indices.shape[0]
        correct += np.sum(predicted_indices == true_indices)

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


# --- K 折交叉验证 (基于 NumPy) ---
def run_k_fold_validation(config, k=5):
    """执行 k 折交叉验证 (NumPy 版本, 支持 MLP 和 CNN)"""
    model_type = config.get('model_type', 'mlp')
    structure_option = config.get('structure_option', 'medium')
    activation_fn = config.get('activation_fn', 'relu')
    regularization_config = configure_regularization(config)
    dropout_rate = regularization_config.get('dropout_rate', 0.0)
    loss_type = config.get('loss_type', 'cross_entropy')
    use_one_hot = loss_type == 'mse'

    # 加载完整训练数据
    temp_config = config.copy()
    temp_config['batch_size'] = 1 # 加载所有数据，后面再分批
    full_loader_info, _ = prepare_data_loaders(temp_config) # 获取原始形状的数据
    all_train_images = full_loader_info['images']
    all_train_labels = full_loader_info['labels'] # 可能是索引或 one-hot
    n_samples = full_loader_info['num_samples']
    batch_size = config.get('batch_size', 64) # 恢复原始 batch_size

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
        print(f'\n--- Fold {fold+1}/{k} ---')

        # 创建当前折的数据加载器信息
        fold_train_loader_info = {
            'images': all_train_images[train_idx],
            'labels': all_train_labels[train_idx],
            'batch_size': batch_size,
            'num_samples': len(train_idx),
            'num_batches': int(np.ceil(len(train_idx) / batch_size)),
            'shuffle': True,
            'use_one_hot': use_one_hot,
            'num_classes': 10
        }
        fold_val_loader_info = {
            'images': all_train_images[val_idx],
            'labels': all_train_labels[val_idx],
            'batch_size': batch_size,
            'num_samples': len(val_idx),
            'num_batches': int(np.ceil(len(val_idx) / batch_size)),
            'shuffle': False,
            'use_one_hot': use_one_hot,
            'num_classes': 10
        }

        # 创建新模型
        model = create_model(model_type, structure_option, dropout_rate, activation_fn)

        # 训练模型
        fold_config = config.copy() # 传递完整的配置
        model, _ = train_model(model, fold_train_loader_info, fold_val_loader_info, fold_config)

        # 在验证集上测试
        accuracy = test_model(model, fold_val_loader_info, loss_type=loss_type)
        fold_accuracies.append(accuracy)
        print(f'Fold {fold+1} Validation Accuracy: {accuracy:.2f}%')

    avg_accuracy = np.mean(fold_accuracies)
    print(f'\nAverage K-Fold Validation Accuracy ({k} folds): {avg_accuracy:.2f}%')
    return avg_accuracy, fold_accuracies


# --- 贝叶斯优化和正则化比较 (占位符) ---
# 这些函数需要外部库 (bayes_opt) 并且逻辑复杂，暂不详细实现 NumPy 版本
def objective_function(learning_rate, momentum, dropout_rate_config, structure_option_idx, activation_fn_idx):
    """优化目标函数 - 用于贝叶斯优化 (需要调用 NumPy 版本的 k-fold)"""
    print("\nRunning Bayesian Optimization Trial...")
    # --- 将浮点参数映射回选项 ---
    structure_options = ['small', 'medium', 'large']
    activation_options = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
    structure_option = structure_options[min(int(structure_option_idx), len(structure_options)-1)]
    activation_fn = activation_options[min(int(activation_fn_idx), len(activation_options)-1)]
    # Dropout 需要根据传入的配置值决定是否启用
    use_dropout = dropout_rate_config > 0.05 # 假设大于 0.05 才启用
    dropout_rate = dropout_rate_config if use_dropout else 0.0
    reg_methods = ['dropout'] if use_dropout else []

    # --- 构建 NumPy 兼容配置 ---
    # (假设 base_config 包含了 model_type, data_dir, num_epochs, batch_size 等固定参数)
    global base_config_for_bayesopt # 需要一个全局或闭包变量来传递固定配置
    if base_config_for_bayesopt is None:
         raise ValueError("需要设置 base_config_for_bayesopt")

    cv_config = base_config_for_bayesopt.copy()
    cv_config.update({
        'learning_rate': float(learning_rate),
        'momentum': float(momentum),
        'structure_option': structure_option,
        'activation_fn': activation_fn,
        'regularization_methods': reg_methods,
        'dropout_rate': dropout_rate,
        'early_stopping': False, # 优化时不启用早停，以获得完整轮数的表现
    })
    # K-Fold 会自行处理 loss_type 和 one-hot
    print(f"Testing config: LR={cv_config['learning_rate']:.4f}, Momentum={cv_config['momentum']:.3f}, Structure={structure_option}, Activation={activation_fn}, Dropout={dropout_rate:.3f}")

    try:
        # 调用 NumPy 版本的 k-fold
        avg_accuracy, _ = run_k_fold_validation(cv_config, k=3) # 使用 3 折加速优化
        print(f"Trial result (Avg Accuracy): {avg_accuracy:.2f}%")
        return avg_accuracy # 贝叶斯优化需要最大化目标
    except Exception as e:
         print(f"错误发生在 K-Fold 验证中: {e}")
         return 0.0 # 返回一个低值表示失败

# 全局变量用于传递基础配置给目标函数
base_config_for_bayesopt = None

def run_bayesian_optimization(config, init_points=5, n_iter=10):
    """运行贝叶斯优化 (NumPy 版本)"""
    global base_config_for_bayesopt
    base_config_for_bayesopt = config.copy() # 存储基础配置

    try:
        from bayes_opt import BayesianOptimization
    except ImportError:
        print("错误: 需要安装 bayesian-optimization 库才能运行贝叶斯优化。")
        print("请运行: pip install bayesian-optimization")
        return None

    # 定义超参数搜索空间
    pbounds = {
        'learning_rate': (1e-4, 1e-1),     # 学习率范围
        'momentum': (0.0, 0.99),           # 动量范围
        'dropout_rate_config': (0.0, 0.7), # Dropout 比率 (0-0.05 表示禁用)
        'structure_option_idx': (0, 2.99), # 对应 small, medium, large
        'activation_fn_idx': (0, 3.99)     # 对应 relu, sigmoid, tanh, leaky_relu
    }

    optimizer = BayesianOptimization(
        f=objective_function, # 使用适配后的 NumPy 目标函数
        pbounds=pbounds,
        random_state=42,
        verbose=2 # verbose = 2 prints details, = 1 prints only when a maximum is observed, = 0 is silent
    )

    print("\n开始贝叶斯优化...")
    # 运行优化
    # acq='ei', xi=0.01 稍微增加探索性
    optimizer.maximize(
        init_points=init_points, # 初始随机点数
        n_iter=n_iter,           # 迭代次数
        acq='ei',                # 采集函数 (Expected Improvement)
        xi=0.01                  # EI 的探索-利用权衡参数
    )

    print("\n贝叶斯优化完成。")
    print("最佳参数:")
    best_params_raw = optimizer.max['params']
    print(best_params_raw)

    # --- 将优化结果转换回可用的配置格式 ---
    best_config = base_config_for_bayesopt.copy()
    structure_options = ['small', 'medium', 'large']
    activation_options = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
    structure_idx = min(int(best_params_raw['structure_option_idx']), len(structure_options)-1)
    activation_idx = min(int(best_params_raw['activation_fn_idx']), len(activation_options)-1)
    dropout_rate_config = best_params_raw['dropout_rate_config']
    use_dropout = dropout_rate_config > 0.05
    dropout_rate = dropout_rate_config if use_dropout else 0.0
    reg_methods = ['dropout'] if use_dropout else []

    best_config.update({
        'learning_rate': best_params_raw['learning_rate'],
        'momentum': best_params_raw['momentum'],
        'structure_option': structure_options[structure_idx],
        'activation_fn': activation_options[activation_idx],
        'regularization_methods': reg_methods,
        'dropout_rate': dropout_rate,
    })
    print("\n转换后的最佳配置:")
    print(best_config)

    # 清理全局变量
    base_config_for_bayesopt = None
    return best_config

def compare_regularization_methods(best_params, config):
    """比较正则化方法 (NumPy 版本)"""
    print("\n--- 开始比较正则化方法 ---")
    base_config = config.copy()
    base_config.update(best_params) # 使用找到的最佳超参数

    results = {}

    # 1. 无正则化
    print("\n测试: 无正则化")
    config_no_reg = base_config.copy()
    config_no_reg['regularization_methods'] = []
    config_no_reg['dropout_rate'] = 0.0
    config_no_reg['regularization_config'] = configure_regularization(config_no_reg)
    train_loader, test_loader = prepare_data_loaders(config_no_reg)
    model_no_reg = create_model(config_no_reg['model_type'], config_no_reg['structure_option'],
                                config_no_reg['dropout_rate'], config_no_reg['activation_fn'])
    model_no_reg, _ = train_model(model_no_reg, train_loader, None, config_no_reg)
    acc_no_reg = test_model(model_no_reg, test_loader, loss_type=config_no_reg['loss_type'])
    results['无正则化'] = acc_no_reg

    # 2. Dropout
    print("\n测试: Dropout")
    config_dropout = base_config.copy()
    config_dropout['regularization_methods'] = ['dropout']
    # 使用优化找到的 dropout rate，如果原本是 0，则用一个默认值如 0.3
    if config_dropout.get('dropout_rate', 0) <= 0.05:
        config_dropout['dropout_rate'] = 0.3
        print("警告: 优化得到的 dropout rate 较低，使用默认值 0.3 进行比较")
    config_dropout['regularization_config'] = configure_regularization(config_dropout)
    train_loader, test_loader = prepare_data_loaders(config_dropout) # 需要重新加载，因为 dropout 配置变了
    model_dropout = create_model(config_dropout['model_type'], config_dropout['structure_option'],
                                 config_dropout['dropout_rate'], config_dropout['activation_fn'])
    model_dropout, _ = train_model(model_dropout, train_loader, None, config_dropout)
    acc_dropout = test_model(model_dropout, test_loader, loss_type=config_dropout['loss_type'])
    results['Dropout'] = acc_dropout

    # 3. 早停 (需要 K-Fold 或单独的验证集) - 这里简化，直接训练然后测试
    # 注意：没有 K-Fold 的早停效果可能不佳，且需要 patience 参数
    # print("\n测试: 早停 (简化)")
    # config_early_stop = base_config.copy()
    # config_early_stop['regularization_methods'] = ['early_stopping']
    # config_early_stop['patience'] = 5 # 使用默认耐心值
    # config_early_stop['regularization_config'] = configure_regularization(config_early_stop)
    # # 早停需要在 train_model 时传入验证集，这里为了简化比较，不单独划分验证集
    # # 因此，这个比较可能不太公平或准确
    # train_loader, test_loader = prepare_data_loaders(config_early_stop)
    # model_early_stop = create_model(config_early_stop['model_type'], config_early_stop['structure_option'],
    #                               config_early_stop.get('dropout_rate', 0.0), config_early_stop['activation_fn'])
    # print("警告: 早停比较未使用独立的验证集，结果可能不准确。")
    # model_early_stop, _ = train_model(model_early_stop, train_loader, None, config_early_stop) # 传入 None 验证集
    # acc_early_stop = test_model(model_early_stop, test_loader, loss_type=config_early_stop['loss_type'])
    # results['早停 (简化)'] = acc_early_stop
    print("\n早停比较需要独立的验证集或 K-Fold 设置，此处跳过。")

    print("\n--- 正则化方法比较结果 ---")
    for method, acc in results.items():
        print(f"{method}: {acc:.2f}%")

    return results


# --- 主函数和用户交互 ---
def main():
    print("\n====== 神经网络训练配置 (NumPy 版本) ======")

    # --- 获取用户选择 ---
    model_options = ['mlp', 'cnn']
    model_type = get_user_choice("选择模型类型:", model_options)

    # run_modes = ['使用推荐配置运行', '手动配置', '使用贝叶斯优化寻找最佳配置', '比较正则化方法']
    run_modes = ['使用推荐配置运行', '手动配置', '使用贝叶斯优化寻找最佳配置'] # 暂时移除正则化比较入口
    run_mode = get_user_choice("选择运行模式:", run_modes)

    # --- 创建基本配置 ---
    config = {
        'model_type': model_type,
        'data_dir': './mnist_data', # 默认数据目录
        'device': 'cpu', # NumPy 只能用 CPU
    }
    config['data_dir'] = get_user_input_path("输入 MNIST 数据集目录:", config['data_dir'])

    # --- 根据运行模式填充或获取配置 ---
    best_params = None # 用于存储贝叶斯优化结果

    if run_mode == '使用推荐配置运行':
        if model_type == 'mlp':
            config.update({
                'structure_option': 'medium', 'activation_fn': 'sigmoid',
                'loss_type': 'cross_entropy', 'optimizer_type': 'sgd',
                'learning_rate': 0.1, 'momentum': 0.9, 'num_epochs': 15, 'batch_size': 64,
                'regularization_methods': ['dropout'], 'dropout_rate': 0.3,
                'early_stopping': False, 'patience': 5
            })
        elif model_type == 'cnn':
             config.update({
                'structure_option': 'medium', 'activation_fn': 'relu', # CNN 常用 ReLU
                'loss_type': 'cross_entropy', 'optimizer_type': 'sgd',
                'learning_rate': 0.05, 'momentum': 0.9, 'num_epochs': 10, 'batch_size': 64, # CNN 可能收敛更快
                'regularization_methods': ['dropout'], 'dropout_rate': 0.4, # CNN 可能需要更高 dropout
                'early_stopping': False, 'patience': 5
             })
        print("\n使用推荐配置:")
        print(config)

    elif run_mode == '使用贝叶斯优化寻找最佳配置':
        print("\n为贝叶斯优化设置基础参数:")
        config['num_epochs'] = get_user_int("输入 K-Fold 训练轮数 (用于评估):", 5) # 优化时减少轮数
        config['batch_size'] = get_user_int("输入批量大小:", 64)
        config['loss_type'] = get_user_choice("选择损失函数:", ['cross_entropy', 'mse'])
        config['optimizer_type'] = 'sgd' # 优化器固定为 SGD
        # 其他固定参数如 data_dir 已设置

        init_points = get_user_int("输入贝叶斯优化初始点数:", 3)
        n_iter = get_user_int("输入贝叶斯优化迭代次数:", 7)

        best_params = run_bayesian_optimization(config, init_points, n_iter)
        if best_params:
             print("\n找到的最佳配置将用于后续训练。")
             config.update(best_params) # 更新 config 为找到的最佳参数
             # 可能需要重置训练轮数
             config['num_epochs'] = get_user_int("输入使用最佳配置的最终训练轮数:", 15)
        else:
             print("贝叶斯优化失败或未找到结果。")
             return # 提前退出

    # elif run_mode == '比较正则化方法':
    #     # 需要先运行贝叶斯优化或手动指定基础参数
    #     print("\n比较正则化方法需要一组基础超参数。")
    #     choice = get_user_choice("从哪里获取基础参数?", ["运行贝叶斯优化", "手动输入"])
    #     if choice == "运行贝叶斯优化":
    #          config['num_epochs'] = get_user_int("输入 K-Fold 训练轮数 (用于评估):", 5)
    #          config['batch_size'] = get_user_int("输入批量大小:", 64)
    #          config['loss_type'] = get_user_choice("选择损失函数:", ['cross_entropy', 'mse'])
    #          init_points = get_user_int("输入贝叶斯优化初始点数:", 3)
    #          n_iter = get_user_int("输入贝叶斯优化迭代次数:", 7)
    #          best_params = run_bayesian_optimization(config, init_points, n_iter)
    #          if not best_params:
    #              print("贝叶斯优化失败，无法比较。")
    #              return
    #     else: # 手动输入基础参数用于比较
    #          best_params = {}
    #          structure_options = ['small', 'medium', 'large']
    #          best_params['structure_option'] = get_user_choice(f"选择 {model_type.upper()} 结构:", structure_options)
    #          activation_options = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
    #          best_params['activation_fn'] = get_user_choice("选择激活函数:", activation_options)
    #          best_params['learning_rate'] = get_user_float("输入学习率:", 0.01)
    #          best_params['momentum'] = get_user_float("输入动量:", 0.9)
    #          config['loss_type'] = get_user_choice("选择损失函数:", ['cross_entropy', 'mse'])
    #          config['optimizer_type'] = 'sgd'
    #          config['num_epochs'] = get_user_int("输入训练轮数:", 10)
    #          config['batch_size'] = get_user_int("输入批量大小:", 64)
    #
    #     compare_regularization_methods(best_params, config)
    #     print("\n程序执行完成。")
    #     return # 比较后退出

    else: # 手动配置
        print("\n手动配置参数:")
        structure_options = ['small', 'medium', 'large']
        config['structure_option'] = get_user_choice(f"选择 {model_type.upper()} 结构大小:", structure_options)
        activation_options = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
        config['activation_fn'] = get_user_choice("选择激活函数:", activation_options)
        loss_options = ['cross_entropy', 'mse']
        config['loss_type'] = get_user_choice("选择损失函数:", loss_options)
        optimizer_options = ['sgd']
        config['optimizer_type'] = get_user_choice("选择优化器:", optimizer_options)

        config['learning_rate'] = get_user_float("输入学习率:", 0.01)
        config['num_epochs'] = get_user_int("输入训练轮数:", 10)
        config['batch_size'] = get_user_int("输入批量大小:", 64)
        if config['optimizer_type'] == 'sgd':
            config['momentum'] = get_user_float("输入动量 (0 表示不用):", 0.9)

        reg_methods = []
        use_dropout = get_user_choice("使用 Dropout?", ['是', '否'])
        if use_dropout == '是':
            reg_methods.append('dropout')
            config['dropout_rate'] = get_user_float("输入 Dropout 比率:", 0.3)
        else:
             config['dropout_rate'] = 0.0

        use_early_stopping = get_user_choice("使用早停? (需要K-Fold或独立验证集，否则无效):", ['是', '否'])
        if use_early_stopping == '是':
            reg_methods.append('early_stopping')
            config['patience'] = get_user_int("输入早停耐心值:", 5)
        else:
            config['early_stopping'] = False

        config['regularization_methods'] = reg_methods

    # --- 配置最终处理和运行 ---
    # 确保 loss_type 和 use_one_hot 匹配
    config['use_one_hot'] = config['loss_type'].lower() == 'mse'
    # 更新正则化配置字典
    config['regularization_config'] = configure_regularization(config)

    # 加载数据
    print("\n加载数据...")
    train_loader_info, test_loader_info = prepare_data_loaders(config)

    # 创建模型
    print("创建模型...")
    model = create_model(
        config['model_type'],
        config.get('structure_option', 'medium'),
        config['regularization_config'].get('dropout_rate', 0.0),
        config.get('activation_fn', 'relu')
    )

    # 训练模型 (手动配置和推荐配置模式下，不使用验证集进行早停)
    # 如果需要早停，需要修改这里，例如划分一部分训练数据作为验证集
    print("\n开始训练...")
    use_val_for_train = config['regularization_config'].get('early_stopping', False)
    if use_val_for_train:
        print("警告: 早停已启用，但未使用独立的验证集或 K-Fold，可能效果不佳。")
        # 可以考虑在这里拆分 train_loader_info 来创建一个 val_loader_info
        # ... 拆分逻辑 ...
        # model, history = train_model(model, train_subset_loader, val_subset_loader, config)
        model, history = train_model(model, train_loader_info, None, config) # 暂不拆分
    else:
        model, history = train_model(model, train_loader_info, None, config)

    # 测试模型
    print("\n开始测试...")
    accuracy = test_model(model, test_loader_info, loss_type=config['loss_type'])
    print(f"\n最终测试准确率: {accuracy:.2f}%")

    print("\n程序执行完成。")


# --- 用户交互辅助函数 ---
def get_user_choice(prompt, options):
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
    while True:
        try:
            default_str = f" (默认: {default})" if default is not None else ""
            val_str = input(f"{prompt}{default_str}: ")
            if not val_str and default is not None:
                return default
            return float(val_str)
        except ValueError:
            print("请输入有效的数字。")

def get_user_int(prompt, default=None):
    while True:
        try:
            default_str = f" (默认: {default})" if default is not None else ""
            val_str = input(f"{prompt}{default_str}: ")
            if not val_str and default is not None:
                return default
            return int(val_str)
        except ValueError:
            print("请输入有效的整数。")

def get_user_input_path(prompt, default=None):
    while True:
        default_str = f" (默认: {default})" if default is not None else ""
        path_str = input(f"{prompt}{default_str}: ")
        path_to_check = path_str if path_str else default

        if path_to_check and os.path.exists(path_to_check) and os.path.isdir(path_to_check):
            # 检查必要的 MNIST 文件是否存在
            required_files = [
                "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
            ]
            files_exist = all(os.path.exists(os.path.join(path_to_check, f)) for f in required_files)
            if files_exist:
                print(f"使用有效的数据路径: {path_to_check}")
                return path_to_check
            else:
                 print(f"错误: 路径 '{path_to_check}' 中缺少 MNIST 文件。请确保包含 {required_files}")
        else:
             print(f"错误: 路径 '{path_to_check}' 不存在或不是一个有效的目录。请重新输入。")


if __name__ == '__main__':
     main()

