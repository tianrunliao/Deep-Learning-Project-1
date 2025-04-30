import numpy as np
import pandas as pd
import os
import sys
import time
import argparse

# 从 NumPy 版本的 function.py 导入
from function import (
    create_model, prepare_data_loaders, train_model, test_model,
    get_batch # 需要 get_batch 来处理数据加载信息
)

# 设置结果保存路径
RESULTS_DIR = "配置测试结果_NumPy" # 区分 NumPy 结果
os.makedirs(RESULTS_DIR, exist_ok=True)

# 配置参数 (移除了 CNN 相关)
MODEL_TYPES = ['mlp'] # 只支持 MLP
STRUCTURE_OPTIONS = ['small', 'medium', 'large']
ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
LOSS_TYPES = ['cross_entropy', 'mse']
OPTIMIZER_TYPES = ['sgd'] # 只支持 SGD
MOMENTUM_VALUES = [0.0, 0.9] # 包含 0 动量
REGULARIZATION_METHODS = [
    [],  # 无正则化
    ['dropout'],
]

# 设置固定参数 (CPU-only)
def get_fixed_config():
    """获取基本配置 (CPU)"""
    return {
        'num_epochs': 3,  # 小规模测试只需3轮
        'batch_size': 64, # CPU batch size
        'learning_rate': 0.1, # NumPy 可能需要调整学习率
        'dropout_rate': 0.3,
        'patience': 3, # 早停暂不使用
        'device': 'cpu', # 固定为 CPU
    }

def get_data_dir():
    """获取数据目录路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 尝试在脚本同级目录下查找 dataset/MNIST
    data_dir_local = os.path.join(script_dir, 'dataset', 'MNIST')
    if os.path.exists(data_dir_local):
        return data_dir_local
    # 尝试在脚本上一级目录下查找 dataset/MNIST
    data_dir_parent = os.path.join(os.path.dirname(script_dir), 'dataset', 'MNIST')
    if os.path.exists(data_dir_parent):
        return data_dir_parent
    # 如果都找不到，返回一个默认或提示错误
    print(f"警告：无法在 {data_dir_local} 或 {data_dir_parent} 找到 MNIST 数据集。请确保数据集存在。")
    return data_dir_parent # 或者可以抛出错误

def generate_model_structure_html(config):
    """生成模型结构的HTML表示 (只处理 MLP)"""
    model_type = config["model_type"]
    structure_option = config["structure_option"]
    activation_fn = config.get("activation_fn", "relu")

    if model_type == "mlp":
        # 获取MLP结构
        if structure_option == "small":
            layers = [784, 128, 10]
        elif structure_option == "medium":
            layers = [784, 256, 128, 10]
        else:  # large
            layers = [784, 512, 256, 128, 10]

        html = '<div class="model-structure">'
        html += '<h5>模型结构示意图</h5>'
        html += '<div class="layers-container">'

        for i, layer_size in enumerate(layers):
            if i == 0: layer_width = 80
            elif i == len(layers) - 1: layer_width = 80
            else: layer_width = 120

            html += f'<div class="layer" style="width:{layer_width}px;">'

            if i == 0: html += '<div class="layer-label">输入层<br>784</div>'
            elif i == len(layers) - 1: html += '<div class="layer-label">输出层<br>10</div>'
            else: html += f'<div class="layer-label">隐藏层{i}<br>{layer_size}</div>'

            if i < len(layers) - 1:
                html += f'<div class="connection"></div>'
                html += f'<div class="activation">{activation_fn}</div>'

            html += '</div>'

        html += '</div></div>'
        return html
    else: # CNN (不再支持)
        return "<p>CNN 结构图不可用 (NumPy 版本未实现)</p>"

def get_config_summary(config):
    """获取配置摘要"""
    summary = []

    # 基本配置
    summary.append(f"模型类型: {config['model_type'].upper()}")
    summary.append(f"结构大小: {config['structure_option']}")
    summary.append(f"激活函数: {config.get('activation_fn', 'relu')}")

    # 训练配置
    summary.append(f"优化器: {config.get('optimizer_type', 'sgd')}")
    summary.append(f"损失函数: {config.get('loss_type', 'cross_entropy')}")

    # 正则化方法
    reg_methods = config.get('regularization_methods', [])
    reg_methods_str = ', '.join(reg_methods) if reg_methods else "无"
    summary.append(f"正则化方法: {reg_methods_str}")
    if 'dropout' in reg_methods:
         summary.append(f"Dropout 率: {config.get('dropout_rate', 0.0)}")

    # 计算设备 (固定为 CPU)
    summary.append(f"计算设备: CPU")

    return summary

def create_subset_loaders_info(train_loader_info, test_loader_info, subset_ratio=0.1):
    """根据 NumPy 数据信息创建训练子集的信息字典，保留完整测试集"""
    # 获取完整训练数据
    full_train_images = train_loader_info['images']
    full_train_labels = train_loader_info['labels']
    dataset_size = train_loader_info['num_samples']

    # 计算子集大小
    subset_size = int(dataset_size * subset_ratio)

    # 随机选择索引
    indices = np.random.permutation(dataset_size)[:subset_size]

    # 创建子集数据
    subset_train_images = full_train_images[indices]
    subset_train_labels = full_train_labels[indices]

    # 创建子集加载器信息字典
    subset_loader_info = train_loader_info.copy()
    subset_loader_info['images'] = subset_train_images
    subset_loader_info['labels'] = subset_train_labels
    subset_loader_info['num_samples'] = subset_size
    subset_loader_info['num_batches'] = int(np.ceil(subset_size / subset_loader_info['batch_size']))

    print(f"创建了训练数据子集信息: 从 {dataset_size} 样本中随机选择 {subset_size} 样本 (比例: {subset_ratio*100:.0f}%)")

    # 测试集信息保持不变
    return subset_loader_info, test_loader_info

def run_test_for_configuration(config, data_loaders_info=None, subset_ratio=0.1):
    """为单个配置运行测试 (NumPy 版本)"""
    try:
        # 获取或创建数据加载器信息
        if data_loaders_info is None:
            # 确保 use_one_hot 在 config 中设置
            config['use_one_hot'] = config['loss_type'] == 'mse'
            full_train_loader_info, full_test_loader_info = prepare_data_loaders(config)
            # 创建子集信息
            train_loader_info, test_loader_info = create_subset_loaders_info(
                full_train_loader_info, full_test_loader_info, subset_ratio
            )
        else:
            train_loader_info, test_loader_info = data_loaders_info

        # 创建模型
        dropout_rate = config['dropout_rate'] if 'dropout' in config.get('regularization_methods', []) else 0.0
        # 确保 dropout_rate 在 config 中，以便 train_model 使用
        config['dropout_rate'] = dropout_rate # 更新 config

        model = create_model(
            config['model_type'],
            config['structure_option'],
            dropout_rate, # 传递给模型构造函数
            config['activation_fn']
        )

        # 配置正则化 (确保传给 train_model)
        config['regularization_config'] = {
             'methods': config.get('regularization_methods', []),
             'dropout_rate': dropout_rate, # 从 config 获取
             'early_stopping': False, # 快速遍历不使用早停
             'patience': config.get('patience', 3)
        }

        # 训练模型 (NumPy 版本)
        print(f"开始训练配置: {config['model_type']} - {config['structure_option']} - {config['activation_fn']} - {config['optimizer_type']} - Reg: {config.get('regularization_methods', [])}")
        start_time = time.time()
        # 注意：NumPy 的 train_model 现在需要 loader_info
        model, history = train_model(model, train_loader_info, None, config, None) # 不传验证集，无回调
        training_time = time.time() - start_time

        # 测试模型 (NumPy 版本)
        accuracy = test_model(model, test_loader_info, config['device'], config['loss_type'])

        # 返回结果
        results = {
            "accuracy": accuracy,
            "training_time": training_time,
            "model_type": config["model_type"],
            "structure_html": generate_model_structure_html(config),
            "config_summary": get_config_summary(config),
        }

        print(f"完成配置: Accuracy: {accuracy:.2f}%")
        return results

    except NotImplementedError as e:
         print(f"跳过配置，因为包含未实现的功能: {e}")
         return {
             "accuracy": 0.0,
             "error": str(e),
             "model_type": config.get("model_type", "未知"),
             "config_summary": get_config_summary(config) if "model_type" in config else ["配置错误"],
             "training_time": 0.0,
         }
    except Exception as e:
        print(f"测试配置时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())

        return {
            "accuracy": 0.0,
            "error": str(e),
            "model_type": config.get("model_type", "未知"),
            "config_summary": get_config_summary(config) if "model_type" in config else ["配置错误"],
             "training_time": 0.0,
        }

def save_results_to_csv(all_results):
    """将所有结果保存为CSV格式"""
    data = []

    for config, results in all_results:
         if results.get('error'): continue # 跳过出错的配置

         row = {
            '模型类型': config['model_type'].upper(),
            '结构大小': config['structure_option'],
            '激活函数': config['activation_fn'],
            '优化器': config['optimizer_type'],
            '损失函数': config['loss_type'],
            '正则化方法': ', '.join(config.get('regularization_methods', [])) if config.get('regularization_methods') else '无',
            '准确率': results.get('accuracy', 0.0)
         }

         if config['optimizer_type'] == 'sgd':
            row['动量'] = config.get('momentum', '-') # 可能没有设置 momentum

         row['训练时间(秒)'] = results.get('training_time', 0)
         row['计算设备'] = 'CPU' # 固定为 CPU

         data.append(row)

    if not data:
        print("没有成功的测试结果可保存到 CSV。")
        return None

    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(RESULTS_DIR, 'all_results_numpy.csv')
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"所有结果已保存到 {csv_path}")
        return csv_path
    except Exception as e:
        print(f"保存 CSV 时出错: {e}")
        return None

def create_index_html(all_results, csv_path):
    """创建包含所有结果的索引HTML页面"""
    html = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>神经网络模型配置测试结果 (NumPy)</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .result-card {{ border: 1px solid #ddd; margin-bottom: 20px; padding: 15px; border-radius: 5px; }}
            .result-card h4 {{ margin-top: 0; color: #333; }}
            .config-summary {{ margin-bottom: 15px; }}
            .config-summary ul {{ padding-left: 20px; }}
            .model-structure {{ margin: 20px 0; }}
            .layers-container {{ display: flex; align-items: center; overflow-x: auto; padding: 10px 0; }}
            .layer {{ text-align: center; padding: 10px; position: relative; min-width: 80px; }}
            .layer-label {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
            .connection {{ height: 2px; background-color: #999; width: 30px; margin: 0 5px; }}
            .activation {{ font-size: 12px; color: #666; margin: 5px 0; }}
            .training-info {{ margin-top: 15px; color: #666; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .accuracy-high {{ color: green; font-weight: bold; }}
            .accuracy-medium {{ color: orange; }}
            .accuracy-low {{ color: red; }}
            .error {{ color: red; font-style: italic;}}
        </style>
    </head>
    <body>
        <h1>神经网络模型配置测试结果 (NumPy 版本)</h1>
        <p>共测试了 <strong>{0}</strong> 种不同的 MLP 配置组合。<a href="{1}">下载完整CSV结果</a></p>

        <h2>排序后的前10个结果 (基于 {2}% 训练数据)</h2>
        <table>
            <tr>
                <th>排名</th>
                <th>模型类型</th>
                <th>结构大小</th>
                <th>激活函数</th>
                <th>优化器</th>
                <th>正则化方法</th>
                <th>准确率</th>
                <th>计算设备</th>
            </tr>
    """.format(
        len([res for _, res in all_results if res.get('accuracy', 0) > 0]), # 只统计成功运行的
        os.path.basename(csv_path) if csv_path else '#',
        int(all_results[0][0].get('subset_ratio', 0.1) * 100) if all_results else 10 # 获取 subset_ratio
    )

    # 按准确率排序 (过滤掉出错的)
    valid_results = [(cfg, res) for cfg, res in all_results if res.get('accuracy') is not None and res.get('error') is None]
    sorted_results = sorted(valid_results, key=lambda x: x[1]['accuracy'], reverse=True)

    # 添加前10个结果
    for i, (config, results) in enumerate(sorted_results[:10]):
        accuracy = results['accuracy']

        # 根据准确率设置CSS类
        if accuracy >= 90: accuracy_class = "accuracy-high" # NumPy 可能准确率较低
        elif accuracy >= 80: accuracy_class = "accuracy-medium"
        else: accuracy_class = "accuracy-low"

        reg_methods = config.get('regularization_methods', [])
        reg_methods_str = ', '.join(reg_methods) if reg_methods else '无'
        device = 'CPU' # 固定

        html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{config['model_type'].upper()}</td>
                <td>{config['structure_option']}</td>
                <td>{config['activation_fn']}</td>
                <td>{config['optimizer_type']}</td>
                <td>{reg_methods_str}</td>
                <td class="{accuracy_class}">{accuracy:.2f}%</td>
                <td>{device}</td>
            </tr>
        """

    html += """
        </table>

        <h2>所有测试配置</h2>
    """

    # 添加每个配置的卡片 (包括出错的)
    for i, (config, results) in enumerate(all_results):
        accuracy = results.get('accuracy', 0.0)
        config_summary = results.get("config_summary", ["配置信息错误"])
        structure_html = results.get("structure_html", "")
        error_msg = results.get("error")

        html += f'<div class="result-card">'
        if error_msg:
             html += f'<h4 class="error">测试配置 #{i+1}: 失败</h4>'
             html += f'<p class="error">错误: {error_msg}</p>'
        else:
             html += f'<h4>测试配置 #{i+1}: 准确率 {accuracy:.2f}%</h4>'

        html += '<div class="config-summary"><h5>配置摘要:</h5><ul>'
        for item in config_summary:
            html += f"<li>{item}</li>"
        html += '</ul></div>'

        if not error_msg: # 只显示成功配置的结构和时间
             html += structure_html
             html += f'<div class="training-info"><p>训练时间: {results.get("training_time", 0):.2f}秒</p></div>'

        html += '</div>'

    html += """
    </body>
    </html>
    """

    # 保存索引HTML文件
    index_path = os.path.join(RESULTS_DIR, 'index_numpy.html')
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"索引HTML已保存到 {index_path}")
        return index_path
    except Exception as e:
        print(f"保存 HTML 时出错: {e}")
        return None

def save_results_to_markdown(all_results):
    """将结果保存为Markdown格式"""
    # 按准确率排序 (过滤错误)
    valid_results = [(cfg, res) for cfg, res in all_results if res.get('accuracy') is not None and res.get('error') is None]
    if not valid_results:
        print("没有有效的测试结果可保存到 Markdown。")
        return None
    sorted_results = sorted(valid_results, key=lambda x: x[1]['accuracy'], reverse=True)

    subset_ratio_percent = int(all_results[0][0].get('subset_ratio', 0.1) * 100) if all_results else 10

    markdown = "# 神经网络模型配置测试结果 (NumPy 版本)\n\n"
    markdown += f"共测试了 **{len(all_results)}** 种不同的 MLP 配置组合 (成功 {len(valid_results)} 种)，使用 **{subset_ratio_percent}%** 的训练数据进行快速评估。\n\n"

    # 添加时间戳
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    markdown += f"测试时间: {current_time}\n\n"

    # 添加表格标题
    markdown += "## 所有成功配置按准确率排序\n\n"
    markdown += "| 排名 | 模型类型 | 结构大小 | 激活函数 | "
    markdown += "优化器 | 损失函数 | 正则化方法 | 准确率 | 训练时间(秒) |\n"
    markdown += "|------|----------|----------|------------|"
    markdown += "--------|----------|------------|---------|---------------|\n"

    # 添加表格内容
    for i, (config, results) in enumerate(sorted_results):
        # 获取配置信息
        model_type = config['model_type'].upper()
        structure = config['structure_option']
        activation = config['activation_fn']
        optimizer = config['optimizer_type']
        loss_type = config['loss_type']

        reg_methods = config.get('regularization_methods', [])
        reg_str = ', '.join(reg_methods) if reg_methods else '无'

        accuracy = results['accuracy']
        training_time = results.get('training_time', 0)

        # 添加行
        markdown += f"| {i+1} | {model_type} | {structure} | {activation} | "
        markdown += f"{optimizer} | {loss_type} | {reg_str} | {accuracy:.2f}% | {training_time:.2f} |\n"

    # 添加失败的配置列表 (可选)
    failed_results = [(cfg, res) for cfg, res in all_results if res.get('error') is not None]
    if failed_results:
         markdown += "\n## 失败的配置\n\n"
         for config, results in failed_results:
              cfg_desc = f"{config.get('model_type','?')}, {config.get('structure_option','?')}, {config.get('activation_fn','?')}, {config.get('optimizer_type','?')}, Loss: {config.get('loss_type','?')}, Reg: {config.get('regularization_methods', [])}"
              markdown += f"- 配置: {cfg_desc}\n"
              markdown += f"  - 错误: {results['error']}\n"

    # 保存文件
    md_path = os.path.join(RESULTS_DIR, 'results_summary_numpy.md')
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        print(f"Markdown结果摘要已保存到 {md_path}")
        return md_path
    except Exception as e:
         print(f"保存 Markdown 时出错: {e}")
         return None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='神经网络模型配置遍历测试 (NumPy)')
    parser.add_argument('--reduced', action='store_true', help='使用减少的配置集进行测试，更快完成')
    parser.add_argument('--subset-ratio', type=float, default=0.1,
                        help='训练数据子集比例 (0.0 to 1.0)，默认10%%')
    return parser.parse_args()

def get_reduced_config_set():
    """返回减少的配置集 (只含 MLP)"""
    return {
        'MODEL_TYPES': ['mlp'],
        'STRUCTURE_OPTIONS': ['medium'],
        'ACTIVATION_FUNCTIONS': ['relu', 'sigmoid'],
        'LOSS_TYPES': ['cross_entropy'],
        'OPTIMIZER_TYPES': ['sgd'],
        'MOMENTUM_VALUES': [0.9], # 减少测试时间
        'REGULARIZATION_METHODS': [[], ['dropout']]
    }

def main():
    """主函数 (NumPy 版本)"""
    # 解析命令行参数
    args = parse_args()
    subset_ratio = args.subset_ratio
    if not 0.0 < subset_ratio <= 1.0:
        print(f"错误: subset_ratio ({subset_ratio}) 必须在 (0.0, 1.0] 之间。")
        sys.exit(1)

    print("====== 开始神经网络模型配置快速遍历测试 (NumPy) ======")
    print(f"将使用 {subset_ratio*100:.0f}% 的训练数据进行快速评估")
    print("所有计算将在 CPU 上运行。")

    # 获取数据目录
    data_dir = get_data_dir()
    print(f"使用数据目录: {data_dir}")

    # 创建基本配置 (CPU)
    base_config = {
        'data_dir': data_dir,
        'subset_ratio': subset_ratio, # 将 subset_ratio 存入 config 以便记录
        **get_fixed_config()
    }

    # 使用减少的配置集
    active_momentum_values = MOMENTUM_VALUES # 默认使用全部
    if args.reduced:
        print("使用减少的配置集进行测试")
        reduced_config = get_reduced_config_set()
        active_model_types = reduced_config['MODEL_TYPES']
        active_structure_options = reduced_config['STRUCTURE_OPTIONS']
        active_activation_functions = reduced_config['ACTIVATION_FUNCTIONS']
        active_loss_types = reduced_config['LOSS_TYPES']
        active_optimizer_types = reduced_config['OPTIMIZER_TYPES']
        active_momentum_values = reduced_config['MOMENTUM_VALUES'] # 使用减少的动量值
        active_regularization_methods = reduced_config['REGULARIZATION_METHODS']
    else:
        active_model_types = MODEL_TYPES
        active_structure_options = STRUCTURE_OPTIONS
        active_activation_functions = ACTIVATION_FUNCTIONS
        active_loss_types = LOSS_TYPES
        active_optimizer_types = OPTIMIZER_TYPES
        active_regularization_methods = REGULARIZATION_METHODS

    # 记录所有配置及其结果
    all_results = []
    total_configs = 0

    # 计算总配置数 (只计算 MLP)
    for model_type in active_model_types:
         if model_type == 'mlp':
             num_sgd_configs = len(active_momentum_values) if 'sgd' in active_optimizer_types else 0
             # 其他优化器暂不支持，所以只考虑 SGD
             num_optimizer_variations = num_sgd_configs

             total_configs += len(active_structure_options) * len(active_activation_functions) * \
                              len(active_loss_types) * num_optimizer_variations * \
                              len(active_regularization_methods)
         # else: # CNN 移除

    print(f"将测试共 {total_configs} 种不同的 MLP 配置，每个配置使用 {subset_ratio*100:.0f}% 的训练数据")

    # 确认是否继续
    if total_configs == 0:
         print("没有有效的配置组合可以测试。请检查配置参数。")
         return
         
    confirm = input("确认开始测试? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消测试")
        return

    # 优化：预先加载数据 (只加载 MLP 需要的数据)
    print("预加载数据以加速测试...")

    mlp_config_base = base_config.copy()
    mlp_config_base['model_type'] = 'mlp'
    # prepare_data_loaders 需要 loss_type 来决定 use_one_hot
    # 我们假设先用 cross_entropy 加载，后面根据需要调整 use_one_hot
    mlp_config_base['loss_type'] = 'cross_entropy'
    mlp_config_base['use_one_hot'] = False

    try:
        mlp_full_train_info, mlp_full_test_info = prepare_data_loaders(mlp_config_base)

        # 创建子集信息
        mlp_subset_train_info, mlp_test_info = create_subset_loaders_info(
            mlp_full_train_info, mlp_full_test_info, subset_ratio
        )

        # 显示数据集信息
        print(f"MLP 训练子集大小: {mlp_subset_train_info['num_samples']} 样本")
        print(f"测试集大小: {mlp_test_info['num_samples']} 样本")
    except Exception as e:
         print(f"加载数据时出错: {e}")
         return

    # 遍历所有可能的配置组合 (只 MLP)
    config_index = 0

    try:
        for model_type in active_model_types: # 实际只有 'mlp'
            if model_type != 'mlp': continue # 跳过非 MLP

            for structure_option in active_structure_options:
                for activation_fn in active_activation_functions:
                    for loss_type in active_loss_types:
                        for optimizer_type in active_optimizer_types: # 实际只有 'sgd'
                            if optimizer_type != 'sgd': continue

                            for reg_methods in active_regularization_methods:
                                 for momentum in active_momentum_values: # 循环动量值

                                     # 基本配置
                                     config = base_config.copy()
                                     config['model_type'] = model_type
                                     config['structure_option'] = structure_option
                                     config['activation_fn'] = activation_fn
                                     config['loss_type'] = loss_type
                                     config['optimizer_type'] = optimizer_type
                                     config['regularization_methods'] = reg_methods
                                     config['momentum'] = momentum # 设置动量
                                     config['use_one_hot'] = (loss_type == 'mse') # 根据损失设置

                                     # 准备当前配置的数据加载器信息 (调整 use_one_hot)
                                     current_train_info = mlp_subset_train_info.copy()
                                     current_test_info = mlp_test_info.copy()
                                     current_train_info['use_one_hot'] = config['use_one_hot']
                                     current_test_info['use_one_hot'] = config['use_one_hot']

                                     # 使用预加载的数据加载器信息
                                     results = run_test_for_configuration(
                                         config, # 传递完整配置
                                         (current_train_info, current_test_info),
                                         subset_ratio # 传递以便记录
                                     )
                                     # 将 subset_ratio 加入 config 以便保存
                                     config_saved = config.copy()
                                     config_saved['subset_ratio'] = subset_ratio 
                                     all_results.append((config_saved, results))

                                     config_index += 1
                                     print(f"完成 {config_index}/{total_configs} 种配置")

    except KeyboardInterrupt:
        print("\n用户中断测试。将保存已完成的结果。")

    except Exception as e:
        print(f"\n测试过程中发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("将保存已完成的结果。")

    finally:
        if all_results:
            # 保存结果为Markdown格式
            md_path = save_results_to_markdown(all_results)

            # 保存所有结果为CSV
            csv_path = save_results_to_csv(all_results)

            # 创建索引HTML
            if csv_path: # 只有成功保存了 CSV 才创建 HTML
                index_path = create_index_html(all_results, csv_path)
            else:
                 index_path = None

            print(f"====== 测试完成! ======")
            print(f"结果保存在 {RESULTS_DIR} 目录下")
            if md_path: print(f"Markdown结果摘要: {md_path}")
            if index_path: print(f"索引文件: {index_path}")
            if csv_path: print(f"CSV结果: {csv_path}")
        else:
            print("没有完成任何测试，无法生成结果。")

if __name__ == "__main__":
    main()
