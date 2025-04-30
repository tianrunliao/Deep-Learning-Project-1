import numpy as np
import gzip
import struct
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 移除 torch GPU 设置
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")  # 使用 M1/M2 的 GPU
# else:
#     device = torch.device("cpu")
# print("Using device:", device)
device = 'cpu' # NumPy runs on CPU
print("Using device: CPU (NumPy)")


# --- 导入 NumPy 版 function.py 组件 ---
try:
    from function import (
        load_mnist_data, BasicCNN_NumPy, BasicMLP_NumPy, # 导入 NumPy 模型
        cross_entropy_loss, cross_entropy_loss_derivative, softmax, # 导入 NumPy 损失和 softmax
        SGD_Optimizer, # 导入 NumPy 优化器
        prepare_data_loaders, get_batch, # 导入 NumPy 数据处理
        get_cnn_structure, # 导入 get_cnn_structure
        ConvLayer # 导入 ConvLayer
    )
    print("成功从 function.py (NumPy 版) 导入模块。")
except ImportError as e:
    print(f"无法导入 function.py (NumPy 版): {e}")
    print("请确保 function.py 存在且已更新为 NumPy 版本。")
    exit()
# --------------------------------------

# 加载 MNIST 数据集 (NumPy load_mnist_data 会处理，保留原始函数以防万一，但注释掉)
# def load_mnist_images(file_path):
#     with gzip.open(file_path, 'rb') as f:
#         magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
#         images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
#     return images
#
# def load_mnist_labels(file_path):
#     with gzip.open(file_path, 'rb') as f:
#         magic, num_labels = struct.unpack(">II", f.read(8))
#         labels = np.frombuffer(f.read(), dtype=np.uint8)
#     return labels

# 数据路径 (保持不变)
train_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# --- 使用 NumPy prepare_data_loaders 加载和预处理数据 ---
print("使用 NumPy prepare_data_loaders 加载数据...")
# 配置字典，指定模型类型为 cnn
config_load = {
    'model_type': 'cnn',
    'batch_size': 64,
    'loss_type': 'cross_entropy', # CE loss 使用索引标签
    'data_dir': '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST'
}
try:
    train_loader_info, test_loader_info = prepare_data_loaders(config_load)
    print(f"NumPy 数据加载完成。训练批次数: {train_loader_info['num_batches']}, 测试批次数: {test_loader_info['num_batches']}")
except FileNotFoundError as e:
     print(f"错误: {e}")
     exit()
# 移除旧的 PyTorch 加载和预处理代码
# train_images = load_mnist_images(train_images_path)
# train_labels = load_mnist_labels(train_labels_path)
# test_images = load_mnist_images(test_images_path)
# test_labels = load_mnist_labels(test_labels_path)
# train_images = torch.tensor(train_images, dtype=torch.float32) / 255.0
# test_images = torch.tensor(test_images, dtype=torch.float32) / 255.0
# train_labels = torch.tensor(train_labels, dtype=torch.long)
# test_labels = torch.tensor(test_labels, dtype=torch.long)
# train_images = train_images.unsqueeze(1)
# test_images = test_images.unsqueeze(1)
# test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
# ---------------------------------------------------------

# 定义具有可配置卷积层和 dropout 的 CNN 模型 (使用 NumPy 版本)
# 移除 PyTorch BasicCNN 定义
# class BasicCNN(nn.Module):
#     ... (PyTorch CNN 实现) ...
# 直接使用导入的 BasicCNN_NumPy
print("使用 NumPy 版 BasicCNN_NumPy 模型。")


# 训练函数（NumPy 版本 - 交叉熵）
def train_numpy(model, train_loader_info, val_loader_info, criterion_fn, criterion_deriv_fn, optimizer, num_epochs, regularization=None, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    loss_fn = criterion_fn
    loss_derivative_fn = criterion_deriv_fn

    for epoch in range(num_epochs):
        model.set_training_mode(True)
        running_loss = 0.0
        indices = np.arange(train_loader_info['num_samples'])
        if train_loader_info['shuffle']:
            np.random.shuffle(indices)

        for i in range(train_loader_info['num_batches']):
            batch_images, batch_labels = get_batch(train_loader_info, i, indices)

            # 前向传播
            logits = model.forward(batch_images)
            # 计算 Softmax (移除，损失函数应处理 logits)
            # probs = softmax(logits)
            # 计算损失 (使用 logits)
            loss = loss_fn(logits, batch_labels)
            # 计算梯度 (使用 logits)
            grad_loss = loss_derivative_fn(logits, batch_labels)

            # 反向传播
            model.backward(grad_loss)
            # 更新参数
            optimizer.step() # NumPy 优化器的 step 方法

            running_loss += loss
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{train_loader_info["num_batches"]}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        # 验证逻辑 (如果需要)
        if regularization == 'early_stopping' and val_loader_info:
            model.set_training_mode(False)
            val_loss = 0.0
            for i in range(val_loader_info['num_batches']):
                val_images, val_labels = get_batch(val_loader_info, i) # 验证集不打乱
                val_logits = model.forward(val_images)
                # 移除 softmax，损失函数处理 logits
                # val_probs = softmax(val_logits)
                batch_val_loss = loss_fn(val_logits, val_labels)
                val_loss += batch_val_loss

            epoch_val_loss = val_loss / val_loader_info['num_batches']
            print(f'Validation Loss: {epoch_val_loss:.4f}')
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
    model.set_training_mode(False) # 训练结束后设为评估模式


# 测试函数 (NumPy 版本)
def test_numpy(model, test_loader_info):
    model.set_training_mode(False) # 确保是评估模式
    correct = 0
    total = 0
    for i in range(test_loader_info['num_batches']):
        images, labels_true = get_batch(test_loader_info, i) # 获取原始标签索引

        logits = model.forward(images)
        predicted_indices = np.argmax(logits, axis=1)

        total += labels_true.shape[0]
        correct += np.sum(predicted_indices == labels_true)

    accuracy = 100 * correct / total
    return accuracy

# 定义不同规模卷积层配置 (结构与 NumPy BasicCNN_NumPy 匹配)
conv_filters_candidates = [
    ([16], [128]), # 'small' in function.py uses [16] conv, [32] fc. Let's use [128] fc here.
    ([16, 32], [128]), # 'medium' in function.py uses [16, 32] conv, [64] fc. Let's use [128] fc here.
    ([32, 64], [128]) # 'large' in function.py uses [64, 128] conv, [128, 64] fc. Let's use [32, 64] conv, [128] fc.
]
# Note: The fc_hidden_units definition in BasicCNN_NumPy ([64,]) might need adjustment
# if we want to exactly match the structure intended here. We'll proceed assuming
# the BasicCNN_NumPy in function.py is the target structure. Let's align candidates.
conv_filters_candidates_numpy = [
    ([16], [32]),       # small
    ([16, 32], [64]),    # medium
    ([32, 64], [128, 64])# large - Note: BasicCNN_NumPy currently only supports one list for fc_hidden_units
]

# 目标函数（贝叶斯优化，NumPy 版本）
def objective_numpy(lr, num_epochs, batch_size, k_folds, conv_filters_index, momentum, dropout_rate, pbar=None):
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    k_folds = int(k_folds)
    conv_filters_index = int(conv_filters_index)
    # Use aligned candidates if needed, or stick to the original intent if BasicCNN_NumPy can handle it.
    # Let's use the structure defined in function.py's get_cnn_structure
    # Structure options in function.py: small, medium, large
    structure_options = ['small', 'medium', 'large']
    structure_option = structure_options[min(conv_filters_index, len(structure_options)-1)] # Map index to option

    regularization_types = ['none', 'dropout', 'early_stopping']
    fold_accuracies = {reg: [] for reg in regularization_types}

    # --- NumPy K-Fold Logic ---
    # Need to implement K-Fold using NumPy data directly
    # Load full training data first
    config_kfold = config_load.copy()
    config_kfold['batch_size'] = 1 # Load all data
    try:
        full_train_loader_info, _ = prepare_data_loaders(config_kfold)
    except FileNotFoundError as e:
        print(f"K-Fold Error: {e}")
        return 0.0 # Return low accuracy on error

    all_train_images = full_train_loader_info['images']
    all_train_labels = full_train_loader_info['labels'] # Indices
    n_samples = full_train_loader_info['num_samples']
    batch_size_fold = int(batch_size) # Use the optimized batch size

    np.random.seed(42) # Use NumPy random seed
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
        print(f'Fold {fold+1}/{k_folds}')
        # Create fold loader info
        fold_train_loader_info = {
            'images': all_train_images[train_idx],
            'labels': all_train_labels[train_idx],
            'batch_size': batch_size_fold,
            'num_samples': len(train_idx),
            'num_batches': int(np.ceil(len(train_idx) / batch_size_fold)),
            'shuffle': True,
            'use_one_hot': False, # CE uses indices
            'num_classes': 10
        }
        fold_val_loader_info = {
            'images': all_train_images[val_idx],
            'labels': all_train_labels[val_idx],
            'batch_size': batch_size_fold,
            'num_samples': len(val_idx),
            'num_batches': int(np.ceil(len(val_idx) / batch_size_fold)),
            'shuffle': False,
            'use_one_hot': False, # CE uses indices
            'num_classes': 10
        }

        for reg in regularization_types:
            print(f'Training fold {fold+1} with {reg} regularization')
            # Create model (NumPy)
            # Use structure_option derived from conv_filters_index
            current_dropout_rate = dropout_rate if reg == 'dropout' else 0.0
            model = BasicCNN_NumPy(
                 conv_filters=get_cnn_structure(structure_option)[0], # Get filters from structure
                 fc_hidden_units=get_cnn_structure(structure_option)[1], # Get fc units from structure
                 dropout_rate=current_dropout_rate,
                 activation_fn='relu' # Assuming ReLU for CNN
            )

            # NumPy Loss and Optimizer
            criterion_fn, criterion_deriv_fn = cross_entropy_loss, cross_entropy_loss_derivative
            optimizer_np = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

            # Train (NumPy)
            train_numpy(model, fold_train_loader_info, fold_val_loader_info if reg == 'early_stopping' else None,
                  criterion_fn, criterion_deriv_fn, optimizer_np, num_epochs, regularization=reg, patience=5)

            # Test on validation set (NumPy)
            val_accuracy = test_numpy(model, fold_val_loader_info)
            fold_accuracies[reg].append(val_accuracy)
            print(f'Fold {fold+1} {reg} Validation Accuracy: {val_accuracy:.2f}%')

    # 更新进度条 (一次 objective 调用完成)
    if pbar is not None:
        pbar.update(1)

    avg_val_accuracies = {reg: np.mean(acc) for reg, acc in fold_accuracies.items()}
    for reg, avg_acc in avg_val_accuracies.items():
        print(f'Average {reg} Validation Accuracy: {avg_acc:.2f}%')
    # Return accuracy without regularization for optimization target
    return avg_val_accuracies.get('none', 0.0)


# 可视化卷积核的函数（NumPy 版本）
def visualize_conv_kernels_numpy(model):
    try:
        # Find the first ConvLayer in the model
        first_conv_layer = None
        for layer in model.layers:
            if isinstance(layer, ConvLayer): # Need to import ConvLayer from function? Yes.
                 from function import ConvLayer # Import here or globally
                 first_conv_layer = layer
                 break

        if first_conv_layer is None:
             print("无法在模型中找到 NumPy 卷积层。")
             return

        conv1_weights = first_conv_layer.weights # Shape: [out_channels, in_channels, K, K]
        num_filters = conv1_weights.shape[0]
        print(f"NumPy 卷积核数量: {num_filters}")
        print(f"示例卷积核值 (第一个卷积核):\n{conv1_weights[0, 0]}") # Assuming in_channels=1

        cols = min(num_filters, 8)
        rows = (num_filters + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten() if num_filters > 1 else [axes]

        for i in range(num_filters):
            filter_weights = conv1_weights[i, 0] # Assuming in_channels=1
            # Normalize for display if needed (optional)
            filter_min = filter_weights.min()
            filter_max = filter_weights.max()
            if filter_max > filter_min:
                 display_weights = (filter_weights - filter_min) / (filter_max - filter_min)
            else:
                 display_weights = filter_weights # Keep as is if flat

            axes[i].imshow(display_weights, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i+1}')

        for i in range(num_filters, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
        # 可选：保存图像以防显示失败
        # plt.savefig('conv_kernels_numpy.png')
        # plt.close()

    except NameError as e:
         print(f"可视化 NumPy 卷积核时出错 (可能未导入 ConvLayer): {e}")
    except Exception as e:
        print(f"可视化 NumPy 卷积核时出错: {e}")


# 使用最佳超参数训练最终模型（NumPy 版本）
def train_final_model_numpy(params, pbar=None):
    print("\n使用最佳超参数训练最终 NumPy 模型...")
    np.random.seed(42) # Use NumPy random seed

    lr = params['lr']
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size']) # Use optimized batch size
    # conv_filters_index = int(params['conv_filters_index']) # Now maps to structure_option
    structure_options = ['small', 'medium', 'large']
    structure_option = structure_options[min(int(params['conv_filters_index']), len(structure_options)-1)]

    momentum = params['momentum']
    dropout_rate = params['dropout_rate']
    # conv_filters = conv_filters_candidates_numpy[conv_filters_index] # Use structure option instead

    print(f"选择的结构: {structure_option}")
    print(f"选择的动量: {momentum}")
    print(f"选择的 Dropout 率: {dropout_rate}")

    # Use full data loader info prepared earlier
    # Ensure batch size in loader info matches the optimized one (or reload)
    if train_loader_info['batch_size'] != batch_size:
         print(f"警告: 优化后的 batch size ({batch_size}) 与加载器 ({train_loader_info['batch_size']}) 不同。将重新加载数据。")
         config_final = config_load.copy()
         config_final['batch_size'] = batch_size
         try:
             final_train_loader_info, final_test_loader_info = prepare_data_loaders(config_final)
         except FileNotFoundError as e:
             print(f"错误: {e}")
             return {}, None # Return empty results and no model
    else:
        final_train_loader_info = train_loader_info
        final_test_loader_info = test_loader_info


    regularization_types = ['none', 'dropout', 'early_stopping']
    results = {}
    final_models = {}

    for reg in regularization_types:
        print(f"\n使用 {reg} 正则化运行最终模型")
        current_dropout_rate = dropout_rate if reg == 'dropout' else 0.0
        # Create final model (NumPy)
        from function import get_cnn_structure # Ensure imported
        conv_filters, fc_hidden_units = get_cnn_structure(structure_option)
        model = BasicCNN_NumPy(
            conv_filters=conv_filters,
            fc_hidden_units=fc_hidden_units,
            dropout_rate=current_dropout_rate,
            activation_fn='relu' # Assuming ReLU
        )

        # NumPy Loss and Optimizer
        criterion_fn, criterion_deriv_fn = cross_entropy_loss, cross_entropy_loss_derivative
        optimizer_np = SGD_Optimizer(model, learning_rate=lr, momentum=momentum)

        # Train final model (NumPy)
        # Use final_train_loader_info, and maybe final_val_loader_info for early stopping
        # Using train as val for simplicity/consistency with original script logic for final run
        final_val_loader_info = final_train_loader_info
        train_numpy(model, final_train_loader_info, final_val_loader_info if reg == 'early_stopping' else None,
              criterion_fn, criterion_deriv_fn, optimizer_np, num_epochs, regularization=reg, patience=5)

        # Test final model (NumPy)
        test_accuracy = test_numpy(model, final_test_loader_info)
        results[reg] = test_accuracy
        final_models[reg] = model # Store the trained model
        print(f'最终测试准确率 ({reg}): {test_accuracy:.2f}%')

    # Update progress bar if provided
    if pbar:
        pbar.update(1) # Assume one final training run corresponds to one step

    # Return results and the model trained without regularization for visualization
    return results, final_models.get('none')


def main():
    # 定义贝叶斯优化的超参数边界 (NumPy 版本)
    pbounds = {
        'lr': (0.001, 0.1), # Might need different range for NumPy
        'num_epochs': (5, 15),
        'batch_size': (16, 128),
        'k_folds': (3, 5),
        'conv_filters_index': (0, 2.99), # Map to small, medium, large structure
        'momentum': (0.0, 0.99), # Include 0 for NumPy default
        'dropout_rate': (0.0, 0.5)
    }

    # Initialize Bayesian Optimization
    bo_optimizer = BayesianOptimization(f=objective_numpy, pbounds=pbounds, random_state=42, verbose=2)

    # 执行贝叶斯优化
    print("开始 NumPy CNN 贝叶斯优化...")
    # Use tqdm for progress bar
    total_iterations = 5 + 10 # init_points + n_iter
    with tqdm(total=total_iterations, desc="Bayesian Optimization") as pbar:
        # Pass the pbar to the objective function (needs modification in objective_numpy)
        # For now, we just update it after maximize finishes. A better way is needed.
        bo_optimizer.maximize(init_points=5, n_iter=10)
        # pbar might not be accurate here if maximize doesn't call objective exactly init+n_iter times due to internal logic.

    # 打印最佳超参数
    print("\n=== NumPy CNN 最佳超参数 ===")
    best_params = bo_optimizer.max['params']
    print(f"最佳平均验证准确率 (无正则化): {bo_optimizer.max['target']:.2f}%")
    print(f"最佳参数: {best_params}")

    # 使用最佳超参数训练最终模型并比较正则化
    # Use tqdm for final training progress (1 step per regularization type)
    with tqdm(total=1, desc="Final Training") as pbar_final:
         final_results, final_model_for_viz = train_final_model_numpy(best_params, pbar=pbar_final)

    # 打印最终结果表格
    df = pd.DataFrame(list(final_results.items()), columns=['Regularization', 'Test Accuracy (%)'])
    print("\n=== NumPy CNN 最终结果表格 ===")
    print(df)

    # 可视化最终模型（无正则化）的卷积核
    if final_model_for_viz:
        print("\n可视化最终模型 (无正则化) 的第一个卷积层卷积核...")
        visualize_conv_kernels_numpy(final_model_for_viz)

if __name__ == '__main__':
    main()