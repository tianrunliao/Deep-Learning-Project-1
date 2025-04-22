import numpy as np
import gzip
import os # 导入os库用于路径操作
import struct # 导入struct库用于解析二进制数据
from scipy.ndimage import shift, rotate, zoom

def load_mnist_idx(path):
    """加载 MNIST IDX 文件 (图像或标签)"""
    with gzip.open(path, 'rb') as f:
        # 读取头部信息
        # >ii: 大端字节序 (>)，两个整数 (ii)
        magic, num_items = struct.unpack(">ii", f.read(8))

        if magic == 2051: # 图像文件
            # >ii: 读取行数和列数
            num_rows, num_cols = struct.unpack(">ii", f.read(8))
            print(f"加载图像: {num_items} 项, {num_rows}x{num_cols}")
            # 读取所有像素数据
            data = np.frombuffer(f.read(), dtype=np.uint8)
            # 重塑为 (N, H, W)
            data = data.reshape(num_items, num_rows, num_cols)
        elif magic == 2049: # 标签文件
            print(f"加载标签: {num_items} 项")
            # 读取所有标签数据
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"无法识别的 MNIST magic number: {magic} in file {path}")
    return data

# 假设 images 是一个形状为 (N, 28, 28) 的MNIST图像数组
def augment_data(images):
    augmented_images = []
    
    for image in images:
        # 平移：随机移动1-2个像素
        shift_x, shift_y = np.random.randint(-2, 3, size=2)
        shifted_image = shift(image, [shift_x, shift_y], mode='nearest')
        augmented_images.append(shifted_image)
        
        # 旋转：随机旋转-10到10度
        angle = np.random.uniform(-10, 10)
        rotated_image = rotate(image, angle, reshape=False, mode='nearest')
        augmented_images.append(rotated_image)
        
        # 缩放：随机缩放0.9到1.1倍
        scale = np.random.uniform(0.9, 1.1)
        # 保持输出图像大小与输入一致
        original_shape = image.shape
        # 使用 order=1 (双线性插值) 可能比 nearest 效果更好，并且需要浮点数输入
        scaled_image_temp = zoom(image.astype(float), scale, mode='nearest', order=1)

        # 计算填充或裁剪量以恢复原始尺寸
        delta_h = scaled_image_temp.shape[0] - original_shape[0]
        delta_w = scaled_image_temp.shape[1] - original_shape[1]

        if scale < 1.0:
            # 缩小时，需要填充
            # delta 是负数，填充量应该是正数
            pad_top = -delta_h // 2
            pad_bottom = -delta_h - pad_top
            pad_left = -delta_w // 2
            pad_right = -delta_w - pad_left
            # 注意：填充值应与图像背景或归一化后的背景值匹配，这里假设为0
            scaled_image = np.pad(scaled_image_temp, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        else:
            # 放大时，需要裁剪
            # delta 是正数
            crop_top = delta_h // 2
            crop_bottom = delta_h - crop_top
            crop_left = delta_w // 2
            crop_right = delta_w - crop_left
            # 确保裁剪索引不为负 (虽然 delta 为正时通常不会，但保险起见)
            start_row = max(0, crop_top)
            end_row = scaled_image_temp.shape[0] - max(0, crop_bottom)
            start_col = max(0, crop_left)
            end_col = scaled_image_temp.shape[1] - max(0, crop_right)
            scaled_image = scaled_image_temp[start_row:end_row, start_col:end_col]

        # 确保最终尺寸完全匹配
        if scaled_image.shape != original_shape:
             scaled_image = zoom(scaled_image, (original_shape[0]/scaled_image.shape[0], original_shape[1]/scaled_image.shape[1]), mode='nearest', order=1)
             # 强制裁剪以防万一
             scaled_image = scaled_image[:original_shape[0], :original_shape[1]]

        augmented_images.append(scaled_image)
    
    return np.array(augmented_images)

def adjust_and_save_datasets(original_images_path, original_labels_path):
    """
    加载 MNIST IDX 格式的图像和标签数据集，归一化并增强图像，调整标签，
    并将它们保存到带有 '_adjusted' 后缀的新路径（以 .npy.gz 格式）。
    """
    print(f"开始处理: {original_images_path} 和 {original_labels_path}")
    try:
        # 加载原始 IDX 图像并归一化
        original_images = load_mnist_idx(original_images_path)
        # 归一化到 [0, 1] 范围
        original_images = original_images.astype(np.float32) / 255.0
        print(f"从以下路径加载并归一化原始图像: {original_images_path}, 形状: {original_images.shape}")

        # 加载原始 IDX 标签
        original_labels = load_mnist_idx(original_labels_path)
        print(f"从以下路径加载原始标签: {original_labels_path}, 形状: {original_labels.shape}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件。请检查路径: {original_images_path} 或 {original_labels_path}")
        return
    except Exception as e:
        print(f"加载原始 IDX 数据集时出错: {e}")
        return

    # 增强图像
    print("开始增强图像...")
    augmented_images = augment_data(original_images) # 形状 (3*N, H, W)
    print(f"增强后的图像形状: {augmented_images.shape}")

    # 调整标签 (为每个增强类型复制标签)
    num_augmentations_per_image = len(augmented_images) // len(original_images)
    if len(augmented_images) % len(original_images) != 0:
        print("警告: 增强后的图像数量不是原始数量的整数倍。标签调整可能不正确。")
        # 可以选择停止执行或继续，这里我们选择继续但打印警告

    adjusted_labels = np.repeat(original_labels, num_augmentations_per_image)
    print(f"调整后的标签形状: {adjusted_labels.shape}")

    # 构建新的保存路径 (在原始文件名后添加 _adjusted)
    def get_adjusted_path(original_path):
        directory = os.path.dirname(original_path)
        full_filename = os.path.basename(original_path)

        # 从 IDX 文件名生成 .npy.gz 文件名
        if full_filename.endswith('.idx3-ubyte.gz'):
            base_name = full_filename[:-len('.idx3-ubyte.gz')]
        elif full_filename.endswith('.idx1-ubyte.gz'):
            base_name = full_filename[:-len('.idx1-ubyte.gz')]
        elif full_filename.endswith('.gz'): # 处理其他 .gz
             base_name, _ = os.path.splitext(full_filename)
             if base_name.endswith('.idx3-ubyte') or base_name.endswith('.idx1-ubyte'):
                 base_name, _ = os.path.splitext(base_name)
        else:
            base_name, _ = os.path.splitext(full_filename) # 无扩展名

        # 统一输出扩展名为 .npy.gz
        ext = '.npy.gz'

        # 添加后缀，避免重复添加
        if not base_name.endswith('_adjusted'):
            adjusted_filename = f"{base_name}_adjusted{ext}"
        else:
            # 如果原始基本名已经包含 _adjusted，则不再添加
            adjusted_filename = f"{base_name}{ext}"

        return os.path.join(directory, adjusted_filename)

    adjusted_images_path = get_adjusted_path(original_images_path)
    adjusted_labels_path = get_adjusted_path(original_labels_path)

    # 保存调整后的图像 (格式为 float32)
    try:
        print(f"正在保存增强后的图像到: {adjusted_images_path}")
        with gzip.open(adjusted_images_path, 'wb') as f:
            np.save(f, augmented_images.astype(np.float32)) # 明确保存为 float32
        print("增强后的图像已成功保存。")
    except Exception as e:
        print(f"保存增强图像时出错: {e}")

    # 保存调整后的标签 (格式为 uint8)
    try:
        print(f"正在保存调整后的标签到: {adjusted_labels_path}")
        with gzip.open(adjusted_labels_path, 'wb') as f:
            np.save(f, adjusted_labels.astype(np.uint8)) # 标签通常保存为 uint8
        print("调整后的标签已成功保存。")
    except Exception as e:
        print(f"保存调整标签时出错: {e}")

# --- 主执行部分 ---

# ****** 请将下面的路径替换为你的实际文件路径 ******

# 定义数据集根目录
dataset_dir = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST'

# 第一个数据集 (例如: 训练集)
# 使用实际的 IDX 文件名
original_train_images_path = os.path.join(dataset_dir, 'train-images-idx3-ubyte.gz')
original_train_labels_path = os.path.join(dataset_dir, 'train-labels-idx1-ubyte.gz')
adjust_and_save_datasets(original_train_images_path, original_train_labels_path)

# 第二个数据集 (例如: 测试集或验证集)
# 如果你有测试集文件，取消注释并检查文件名
# original_test_images_path = os.path.join(dataset_dir, 't10k-images-idx3-ubyte.gz') # <-- 检查文件名
# original_test_labels_path = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte.gz') # <-- 检查文件名
# adjust_and_save_datasets(original_test_images_path, original_test_labels_path)

print("\n所有处理完成。")


# 移除旧的示例代码
# # 示例用法
# # 假设 images 是一个形状为 (N, 28, 28) 的MNIST图像数组
# # images = mnist_data['images']  # 假设加载了MNIST数据
#
# # 创建一些示例图像数据 (替换成你实际加载的数据)
# example_images = np.random.rand(10, 28, 28) # 创建10个随机28x28图像
#
# # 增强数据
# augmented_images = augment_data(example_images)
#
# # 定义保存路径
# save_path = '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset adjusted/augmented_images.npy.gz'
#
# # 保存增强后的图像到 .gz 文件
# try:
#     with gzip.open(save_path, 'wb') as f:
#         np.save(f, augmented_images)
#     print(f"增强后的数据已成功保存到: {save_path}")
# except Exception as e:
#     print(f"保存文件时出错: {e}")
#
# # 如何加载保存的数据 (可选)
# # try:
# #     with gzip.open(save_path, 'rb') as f:
# #         loaded_augmented_images = np.load(f)
# #     print(f"从 {save_path} 加载了 {loaded_augmented_images.shape} 的数据")
# # except Exception as e:
# #     print(f"加载文件时出错: {e}")