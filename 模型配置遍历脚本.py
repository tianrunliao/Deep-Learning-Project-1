import torch
import pandas as pd
import os
import sys
import time
import argparse
import random
from function import (
    create_model, prepare_data_loaders, train_model, test_model,
    setup_gpu, print_gpu_memory_usage, clear_gpu_memory # 现在从 function 导入
)
from torch.utils.data import Subset, DataLoader

# 设置结果保存路径
RESULTS_DIR = "配置测试结果"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 配置参数
MODEL_TYPES = ['mlp', 'cnn']
STRUCTURE_OPTIONS = ['small', 'medium', 'large']
ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
POOL_TYPES = ['max', 'avg']  # 仅适用于CNN
LOSS_TYPES = ['cross_entropy', 'mse']
OPTIMIZER_TYPES = ['sgd', 'adam']
MOMENTUM_VALUES = [0.9]  # 仅适用于SGD
USE_AUGMENTED_DATA = [False, True]
REGULARIZATION_METHODS = [
    [],  # 无正则化
    ['dropout'],
    ['early_stopping'],
    ['dropout', 'early_stopping']
]

# 设置固定参数
def get_fixed_config(use_gpu=True):
    """获取基本配置"""
    # 检查GPU可用性
    use_cuda, device = (True, 'cuda') if use_gpu and torch.cuda.is_available() else (False, 'cpu')
    
    return {
        'num_epochs': 3,  # 小规模测试只需3轮
        'batch_size': 64 if use_cuda else 32,  # GPU上使用更大的批量
        'learning_rate': 0.01,
        'dropout_rate': 0.3,
        'patience': 3,
        'device': device,
    }

def get_data_dir():
    """获取数据目录路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'dataset', 'MNIST')
    return data_dir

def generate_model_structure_html(config):
    """生成模型结构的HTML表示，类似于app.py中的函数"""
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
            if i == 0:
                layer_width = 80
            elif i == len(layers) - 1:
                layer_width = 80
            else:
                layer_width = 120
                
            html += f'<div class="layer" style="width:{layer_width}px;">'
            
            if i == 0:
                html += '<div class="layer-label">输入层<br>784</div>'
            elif i == len(layers) - 1:
                html += '<div class="layer-label">输出层<br>10</div>'
            else:
                html += f'<div class="layer-label">隐藏层{i}<br>{layer_size}</div>'
            
            if i < len(layers) - 1:
                html += f'<div class="connection"></div>'
                html += f'<div class="activation">{activation_fn}</div>'
            
            html += '</div>'
        
        html += '</div></div>'
        return html
    else:  # CNN
        pool_type = config.get("pool_type", "max")
        html = '<div class="model-structure">'
        html += '<h5>CNN模型结构示意图</h5>'
        html += '<div class="cnn-container">'
        
        html += '<div class="cnn-layer input-layer"><div class="layer-label">输入<br>28x28x1</div></div>'
        
        conv1_filters = {'small': 16, 'medium': 32, 'large': 64}[structure_option]
        html += f'<div class="cnn-layer conv-layer"><div class="layer-label">卷积层 1<br>{conv1_filters}个3x3滤波器</div></div>'
        html += f'<div class="activation">{activation_fn}</div>'
        
        html += f'<div class="cnn-layer pool-layer"><div class="layer-label">{pool_type}池化<br>14x14x{conv1_filters}</div></div>'
        
        conv2_filters = {'small': 32, 'medium': 64, 'large': 128}[structure_option]
        html += f'<div class="cnn-layer conv-layer"><div class="layer-label">卷积层 2<br>{conv2_filters}个3x3滤波器</div></div>'
        html += f'<div class="activation">{activation_fn}</div>'
        
        html += f'<div class="cnn-layer pool-layer"><div class="layer-label">{pool_type}池化<br>7x7x{conv2_filters}</div></div>'
        
        html += f'<div class="cnn-layer fc-layer"><div class="layer-label">全连接层<br>128</div></div>'
        html += f'<div class="activation">{activation_fn}</div>'
        
        html += '<div class="cnn-layer output-layer"><div class="layer-label">输出层<br>10</div></div>'
        
        html += '</div></div>'
        return html

def get_config_summary(config):
    """获取配置摘要，类似于app.py中的函数"""
    summary = []
    
    # 基本配置
    summary.append(f"模型类型: {config['model_type'].upper()}")
    summary.append(f"结构大小: {config['structure_option']}")
    summary.append(f"激活函数: {config.get('activation_fn', 'relu')}")
    
    # CNN特有配置
    if config['model_type'] == 'cnn':
        summary.append(f"池化类型: {config.get('pool_type', 'max')}")
    
    # 训练配置
    summary.append(f"优化器: {config.get('optimizer_type', 'sgd')}")
    summary.append(f"损失函数: {config.get('loss_type', 'cross_entropy')}")
    summary.append(f"使用增强数据: {'是' if config.get('use_augmented_data', False) else '否'}")
    
    # 正则化方法
    reg_methods = config.get('regularization_methods', [])
    if reg_methods:
        reg_methods_str = ', '.join(reg_methods)
    else:
        reg_methods_str = "无"
    summary.append(f"正则化方法: {reg_methods_str}")
    
    # 计算设备
    device = config.get('device', 'cpu')
    summary.append(f"计算设备: {'GPU' if 'cuda' in device else 'CPU'}")
    
    return summary

def create_subset_loaders(train_loader, test_loader, subset_ratio=0.1):
    """创建训练集子集的数据加载器，保留完整测试集"""
    # 获取训练数据集
    train_dataset = train_loader.dataset
    dataset_size = len(train_dataset)
    
    # 计算子集大小
    subset_size = int(dataset_size * subset_ratio)
    
    # 随机选择索引
    indices = torch.randperm(dataset_size)[:subset_size].tolist()
    
    # 创建子集
    subset = Subset(train_dataset, indices)
    
    # 创建新的DataLoader
    subset_loader = DataLoader(
        subset, 
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0
    )
    
    print(f"创建了训练数据子集: 从 {dataset_size} 样本中随机选择 {subset_size} 样本 (比例: {subset_ratio*100:.0f}%)")
    
    return subset_loader, test_loader

def run_test_for_configuration(config, data_loaders=None, subset_ratio=0.1):
    """为单个配置运行测试"""
    try:
        # 获取训练和测试数据加载器
        if data_loaders is None:
            train_loader, test_loader = prepare_data_loaders(config)
            # 创建子集
            train_loader, test_loader = create_subset_loaders(train_loader, test_loader, subset_ratio)
        else:
            train_loader, test_loader = data_loaders
        
        # 创建模型
        dropout_rate = config['dropout_rate'] if 'dropout' in config.get('regularization_methods', []) else 0.0
        model = create_model(
            config['model_type'],
            config['structure_option'],
            dropout_rate,
            config['activation_fn'],
            config.get('pool_type', 'max')
        )
        
        # 设置进度回调函数
        def progress_callback(epoch, total_epochs):
            print(f"训练进度: {(epoch+1)/total_epochs*100:.0f}%", end='\r')
        
        # 训练模型
        print(f"开始训练配置: {config['model_type']} - {config['activation_fn']} - {config['optimizer_type']}")
        start_time = time.time()
        model, history = train_model(model, train_loader, None, config, progress_callback)
        training_time = time.time() - start_time
        
        # 测试模型
        accuracy = test_model(model, test_loader, config['device'], config['loss_type'])
        
        # 清理GPU内存
        if 'cuda' in config['device']:
            model = model.to('cpu')  # 将模型移到CPU
            del model  # 删除模型
            clear_gpu_memory()  # 清理GPU缓存
        
        # 返回结果
        results = {
            "accuracy": accuracy,
            "training_time": training_time,
            "model_type": config["model_type"],
            "structure_html": generate_model_structure_html(config),
            "config_summary": get_config_summary(config),
        }
        
        print(f"完成配置: {config['model_type']} - {config['activation_fn']} - {config['optimizer_type']} - 准确率: {accuracy:.2f}%")
        return results
    
    except Exception as e:
        print(f"测试配置时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # 清理GPU内存
        if 'device' in config and 'cuda' in config['device']:
            clear_gpu_memory()
            
        return {
            "accuracy": 0.0,
            "error": str(e),
            "model_type": config["model_type"],
            "config_summary": get_config_summary(config),
        }

def save_results_to_csv(all_results):
    """将所有结果保存为CSV格式"""
    data = []
    
    for config, results in all_results:
        row = {
            '模型类型': config['model_type'].upper(),
            '结构大小': config['structure_option'],
            '激活函数': config['activation_fn'],
            '优化器': config['optimizer_type'],
            '准确率': results['accuracy']
        }
        
        if config['model_type'] == 'cnn':
            row['池化类型'] = config.get('pool_type', 'max')
        
        row['损失函数'] = config['loss_type']
        row['使用增强数据'] = '是' if config.get('use_augmented_data', False) else '否'
        
        reg_methods = config.get('regularization_methods', [])
        row['正则化方法'] = ', '.join(reg_methods) if reg_methods else '无'
        
        if config['optimizer_type'] == 'sgd':
            row['动量'] = config.get('momentum', 0.9)
        
        row['训练时间(秒)'] = results.get('training_time', 0)
        row['计算设备'] = 'GPU' if 'cuda' in config.get('device', 'cpu') else 'CPU'
        
        data.append(row)
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(RESULTS_DIR, 'all_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"所有结果已保存到 {csv_path}")
    return csv_path

def create_index_html(all_results, csv_path):
    """创建包含所有结果的索引HTML页面"""
    html = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>神经网络模型配置测试结果</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .result-card {{ border: 1px solid #ddd; margin-bottom: 20px; padding: 15px; border-radius: 5px; }}
            .result-card h4 {{ margin-top: 0; color: #333; }}
            .config-summary {{ margin-bottom: 15px; }}
            .config-summary ul {{ padding-left: 20px; }}
            .model-structure {{ margin: 20px 0; }}
            .layers-container, .cnn-container {{ display: flex; align-items: center; overflow-x: auto; padding: 10px 0; }}
            .layer {{ text-align: center; padding: 10px; position: relative; min-width: 80px; }}
            .layer-label {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
            .connection {{ height: 2px; background-color: #999; width: 30px; margin: 0 5px; }}
            .activation {{ font-size: 12px; color: #666; margin: 5px 0; }}
            .cnn-layer {{ margin: 0 5px; }}
            .training-info {{ margin-top: 15px; color: #666; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .accuracy-high {{ color: green; font-weight: bold; }}
            .accuracy-medium {{ color: orange; }}
            .accuracy-low {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>神经网络模型配置测试结果</h1>
        <p>共测试了 <strong>{0}</strong> 种不同的配置组合。<a href="{1}">下载完整CSV结果</a></p>
        
        <h2>排序后的前10个结果</h2>
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
    """.format(len(all_results), os.path.basename(csv_path))
    
    # 按准确率排序
    sorted_results = sorted(all_results, key=lambda x: x[1]['accuracy'], reverse=True)
    
    # 添加前10个结果
    for i, (config, results) in enumerate(sorted_results[:10]):
        accuracy = results['accuracy']
        
        # 根据准确率设置CSS类
        if accuracy >= 95:
            accuracy_class = "accuracy-high"
        elif accuracy >= 90:
            accuracy_class = "accuracy-medium"
        else:
            accuracy_class = "accuracy-low"
        
        reg_methods = config.get('regularization_methods', [])
        reg_methods_str = ', '.join(reg_methods) if reg_methods else '无'
        device = 'GPU' if 'cuda' in config.get('device', 'cpu') else 'CPU'
        
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
    
    # 添加每个配置的卡片
    for i, (config, results) in enumerate(all_results):
        accuracy = results['accuracy']
        config_summary = results["config_summary"]
        structure_html = results.get("structure_html", "")
        
        html += f"""
        <div class="result-card">
            <h4>测试配置 #{i+1}: 准确率 {accuracy:.2f}%</h4>
            <div class="config-summary">
                <h5>配置摘要:</h5>
                <ul>
        """
        
        for item in config_summary:
            html += f"            <li>{item}</li>\n"
        
        html += f"""
                </ul>
            </div>
            {structure_html}
            <div class="training-info">
                <p>训练时间: {results.get('training_time', 0):.2f}秒</p>
            </div>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    # 保存索引HTML文件
    index_path = os.path.join(RESULTS_DIR, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"索引HTML已保存到 {index_path}")
    return index_path

def save_results_to_markdown(all_results):
    """将结果保存为Markdown格式"""
    # 按准确率排序
    sorted_results = sorted(all_results, key=lambda x: x[1]['accuracy'], reverse=True)
    
    markdown = "# 神经网络模型配置测试结果\n\n"
    markdown += f"共测试了 **{len(all_results)}** 种不同的配置组合，使用10%的训练数据进行快速评估。\n\n"
    
    # 添加时间戳
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    markdown += f"测试时间: {current_time}\n\n"
    
    # 添加表格标题
    markdown += "## 所有配置按准确率排序\n\n"
    markdown += "| 排名 | 模型类型 | 结构大小 | 激活函数 | "
    markdown += "优化器 | 损失函数 | 池化类型 | 正则化方法 | 准确率 | 训练时间(秒) |\n"
    markdown += "|------|----------|----------|------------|"
    markdown += "--------|----------|----------|------------|---------|---------------|\n"
    
    # 添加表格内容
    for i, (config, results) in enumerate(sorted_results):
        # 获取配置信息
        model_type = config['model_type'].upper()
        structure = config['structure_option']
        activation = config['activation_fn']
        optimizer = config['optimizer_type']
        loss_type = config['loss_type']
        pool_type = config.get('pool_type', '-') if model_type == 'CNN' else '-'
        
        reg_methods = config.get('regularization_methods', [])
        reg_str = ', '.join(reg_methods) if reg_methods else '无'
        
        accuracy = results['accuracy']
        training_time = results.get('training_time', 0)
        
        # 添加行
        markdown += f"| {i+1} | {model_type} | {structure} | {activation} | "
        markdown += f"{optimizer} | {loss_type} | {pool_type} | {reg_str} | {accuracy:.2f}% | {training_time:.2f} |\n"
    
    # 保存文件
    md_path = os.path.join(RESULTS_DIR, 'results_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"Markdown结果摘要已保存到 {md_path}")
    return md_path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='神经网络模型配置遍历测试')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU加速计算')
    parser.add_argument('--reduced', action='store_true', help='使用减少的配置集进行测试，更快完成')
    parser.add_argument('--subset-ratio', type=float, default=0.1, help='训练数据子集比例，默认10%')
    return parser.parse_args()

def get_reduced_config_set():
    """返回减少的配置集，用于快速测试"""
    return {
        'MODEL_TYPES': ['mlp', 'cnn'],
        'STRUCTURE_OPTIONS': ['medium'],
        'ACTIVATION_FUNCTIONS': ['relu', 'sigmoid'],
        'POOL_TYPES': ['max'],
        'LOSS_TYPES': ['cross_entropy'],
        'OPTIMIZER_TYPES': ['sgd'],
        'REGULARIZATION_METHODS': [[], ['dropout']]
    }

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    use_gpu = args.use_gpu
    subset_ratio = args.subset_ratio
    
    print("====== 开始神经网络模型配置快速遍历测试 ======")
    print(f"将使用 {subset_ratio*100:.0f}% 的训练数据进行快速评估")
    
    # 如果启用GPU，设置GPU环境
    if use_gpu:
        has_gpu, _ = setup_gpu()
        if not has_gpu:
            print("未检测到可用GPU，将使用CPU运行")
            use_gpu = False
    else:
        print("将使用CPU运行测试")
    
    # 获取数据目录
    data_dir = get_data_dir()
    print(f"使用数据目录: {data_dir}")
    
    # 创建基本配置
    base_config = {
        'data_dir': data_dir,
        **get_fixed_config(use_gpu)
    }
    
    # 使用减少的配置集
    if args.reduced:
        print("使用减少的配置集进行测试")
        reduced_config = get_reduced_config_set()
        active_model_types = reduced_config['MODEL_TYPES']
        active_structure_options = reduced_config['STRUCTURE_OPTIONS']
        active_activation_functions = reduced_config['ACTIVATION_FUNCTIONS']
        active_pool_types = reduced_config['POOL_TYPES']
        active_loss_types = reduced_config['LOSS_TYPES']
        active_optimizer_types = reduced_config['OPTIMIZER_TYPES']
        active_regularization_methods = reduced_config['REGULARIZATION_METHODS']
    else:
        active_model_types = MODEL_TYPES
        active_structure_options = STRUCTURE_OPTIONS
        active_activation_functions = ACTIVATION_FUNCTIONS
        active_pool_types = POOL_TYPES
        active_loss_types = LOSS_TYPES
        active_optimizer_types = OPTIMIZER_TYPES
        active_regularization_methods = REGULARIZATION_METHODS
    
    # 记录所有配置及其结果
    all_results = []
    total_configs = 0
    
    # 计算总配置数
    for model_type in active_model_types:
        if model_type == 'mlp':
            # MLP配置: 结构 x 激活函数 x 损失类型 x 优化器 x 正则化方法
            total_configs += len(active_structure_options) * len(active_activation_functions) * len(active_loss_types) * \
                             len(active_optimizer_types) * len(active_regularization_methods)
        else:  # CNN
            # CNN配置: 结构 x 激活函数 x 池化类型 x 损失类型 x 优化器 x 正则化方法
            total_configs += len(active_structure_options) * len(active_activation_functions) * len(active_pool_types) * \
                             len(active_loss_types) * len(active_optimizer_types) * len(active_regularization_methods)
    
    print(f"将测试共 {total_configs} 种不同配置，每个配置使用 {subset_ratio*100:.0f}% 的训练数据")
    
    # 确认是否继续
    confirm = input("确认开始测试? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消测试")
        return
    
    # 优化：预先加载数据
    print("预加载数据以加速测试...")
    
    mlp_config = base_config.copy()
    mlp_config['model_type'] = 'mlp'
    cnn_config = base_config.copy()
    cnn_config['model_type'] = 'cnn'
    
    mlp_train_loader, mlp_test_loader = prepare_data_loaders(mlp_config)
    cnn_train_loader, cnn_test_loader = prepare_data_loaders(cnn_config)
    
    # 创建子集
    mlp_subset_loader, mlp_test_loader = create_subset_loaders(mlp_train_loader, mlp_test_loader, subset_ratio)
    cnn_subset_loader, cnn_test_loader = create_subset_loaders(cnn_train_loader, cnn_test_loader, subset_ratio)
    
    # 显示数据集信息
    print(f"MLP训练子集大小: {len(mlp_subset_loader.dataset)} 样本")
    print(f"CNN训练子集大小: {len(cnn_subset_loader.dataset)} 样本")
    print(f"测试集大小: {len(mlp_test_loader.dataset)} 样本")
    
    # 遍历所有可能的配置组合
    config_index = 0
    
    try:
        for model_type in active_model_types:
            for structure_option in active_structure_options:
                for activation_fn in active_activation_functions:
                    for loss_type in active_loss_types:
                        for optimizer_type in active_optimizer_types:
                            for reg_methods in active_regularization_methods:
                                
                                # 基本配置
                                config = base_config.copy()
                                config['model_type'] = model_type
                                config['structure_option'] = structure_option
                                config['activation_fn'] = activation_fn
                                config['loss_type'] = loss_type
                                config['optimizer_type'] = optimizer_type
                                config['regularization_methods'] = reg_methods
                                
                                # 如果是SGD，添加动量参数
                                if optimizer_type == 'sgd':
                                    for momentum in MOMENTUM_VALUES:
                                        config['momentum'] = momentum
                                
                                # 如果是CNN，添加池化类型
                                if model_type == 'cnn':
                                    for pool_type in active_pool_types:
                                        config_cnn = config.copy()
                                        config_cnn['pool_type'] = pool_type
                                        
                                        # 使用预加载的数据加载器
                                        results = run_test_for_configuration(
                                            config_cnn, 
                                            (cnn_subset_loader, cnn_test_loader)
                                        )
                                        all_results.append((config_cnn, results))
                                        
                                        config_index += 1
                                        
                                        print(f"完成 {config_index}/{total_configs} 种配置")
                                else:
                                    # 使用预加载的数据加载器
                                    results = run_test_for_configuration(
                                        config, 
                                        (mlp_subset_loader, mlp_test_loader)
                                    )
                                    all_results.append((config, results))
                                    
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
            index_path = create_index_html(all_results, csv_path)
            
            print(f"====== 测试完成! ======")
            print(f"结果保存在 {RESULTS_DIR} 目录下")
            print(f"Markdown结果摘要: {md_path}")
            print(f"索引文件: {index_path}")
            print(f"CSV结果: {csv_path}")
        else:
            print("没有完成任何测试，无法生成结果。")

if __name__ == "__main__":
    main()
