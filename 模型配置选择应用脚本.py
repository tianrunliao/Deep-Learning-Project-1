# 必要导入
import torch
import pandas as pd
import os
import sys
import time
import argparse
import re # 用于解析Markdown

# 获取包含此脚本的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 确保脚本所在目录在Python搜索路径中
# (通常运行脚本时会自动添加，但显式添加更保险)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# 现在尝试直接导入 function 模块
try:
    from function import (
        create_model, prepare_data_loaders, train_model, test_model,
        setup_gpu, print_gpu_memory_usage, clear_gpu_memory
    )
    # 可以加一句打印确认导入成功
    print("成功导入 'function' 模块。")
except ImportError as e:
    print(f"错误：仍然无法导入 function.py 模块。")
    print(f"脚本目录: {script_dir}")
    print(f"Python 搜索路径 (sys.path): {sys.path}")
    print(f"原始错误: {e}")
    sys.exit(1)

# --- 配置和常量 ---
FULL_RESULTS_DIR = "完整数据集测试结果"
os.makedirs(FULL_RESULTS_DIR, exist_ok=True)
DEFAULT_FULL_EPOCHS = 10 # 全数据集训练轮数

# --- 解析 Markdown 结果 ---

def parse_markdown_results(md_filepath):
    """解析results_summary.md文件中的表格，返回配置列表"""
    configs = []
    try:
        with open(md_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_line = None
        separator_index = -1
        header_found = False

        # 1. 查找表头行和分隔符行
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 寻找包含关键字的表头行
            if line.startswith('|') and '排名' in line and '模型类型' in line and not header_found:
                header_line = line
                header_found = True
                print(f"找到潜在表头行: {header_line}")
            
            # 寻找分隔符行，确保它在表头之后
            if header_found and line.startswith('|------|'):
                separator_index = i
                print(f"找到分隔符行于索引: {separator_index}")
                break # 找到表头和分隔符后即可停止初步扫描

        # 2. 验证是否成功找到表头和分隔符
        if not header_line or separator_index == -1:
            print("错误：未能正确识别 Markdown 表格的表头或分隔符。")
            return []

        # 3. 解析表头
        header = [h.strip() for h in header_line.strip('|').split('|') if h.strip()]
        required_cols = ['排名', '模型类型', '结构大小', '激活函数', '优化器', '损失函数', '准确率']
        if not all(col in header for col in required_cols):
            print(f"错误：Markdown文件表头缺失必需列。找到：{header}")
            return []
        header_map = {h: i for i, h in enumerate(header)} # 创建列名到索引的映射
        print(f"成功解析表头: {header}")

        # 4. 解析数据行 (从分隔符之后开始)
        table_started = False
        for line in lines[separator_index + 1:]:
            line = line.strip()
            if not line:
                continue

            if line.startswith('|'):
                parts = [p.strip() for p in line.strip('|').split('|')] 
                if len(parts) >= len(header): # 确保行数据足够长
                     try:
                         # 使用映射安全地访问列
                         config = {
                             'rank': int(parts[header_map['排名']]),
                             'model_type': parts[header_map['模型类型']].lower(),
                             'structure_option': parts[header_map['结构大小']].lower(),
                             'activation_fn': parts[header_map['激活函数']].lower(),
                             'optimizer_type': parts[header_map['优化器']].lower(),
                             'loss_type': parts[header_map['损失函数']].lower(),
                             'pool_type': parts[header_map.get('池化类型', -1)].lower() if '池化类型' in header_map and parts[header_map.get('池化类型')] != '-' else None, # CNN才有，处理'-'
                             'regularization_methods_str': parts[header_map.get('正则化方法', -1)],
                             'accuracy_10_percent': float(parts[header_map['准确率']].rstrip('%')),
                             'training_time_10_percent': float(parts[header_map.get('训练时间(秒)', -1)]) if '训练时间(秒)' in header_map and parts[header_map.get('训练时间(秒)')] != '-' else 0.0
                         }

                         # 解析正则化方法
                         reg_str = config['regularization_methods_str']
                         if reg_str == '无' or not reg_str or reg_str == '-':
                             config['regularization_methods'] = []
                         else:
                             # 简单按逗号分割，去除可能存在的空格
                             config['regularization_methods'] = [m.strip() for m in reg_str.split(',') if m.strip()]

                         # 修正 CNN 池化类型解析
                         if config['model_type'] != 'cnn':
                             config['pool_type'] = None # MLP 没有池化
                         # else: config['pool_type'] 已在上面处理

                         configs.append(config)
                     except (ValueError, IndexError, KeyError) as e:
                          print(f"警告：解析数据行时出错: '{line}' - 错误: {e} - 列索引映射: {header_map}")
                          continue # 跳过格式错误的行
                else:
                     print(f"警告：跳过格式不匹配的数据行: '{line}' (期望至少 {len(header)} 列, 找到 {len(parts)}) ")

        print(f"成功从 Markdown 文件解析了 {len(configs)} 条配置。")
        return configs

    except FileNotFoundError:
        print(f"错误: Markdown 结果文件未找到: {md_filepath}")
        return []
    except Exception as e:
        print(f"错误: 解析 Markdown 文件时发生未知错误: {e}")
        return []

# --- 配置选择逻辑 ---

def select_configs_for_full_training(configs, top_n=20, top_mlp=5, top_sgd=2, top_other_activations=2):
    """根据预定义的策略选择配置进行全数据集训练"""
    if not configs:
        return []

    selected_configs = []
    selected_hashes = set() # 用于去重

    def config_hash(cfg):
        # 创建一个唯一标识符，忽略排名和性能指标
        key_parts = [
            cfg['model_type'], cfg['structure_option'], cfg['activation_fn'],
            cfg['optimizer_type'], cfg['loss_type'],
            cfg.get('pool_type', 'None'), # 处理None
            ','.join(sorted(cfg.get('regularization_methods', []))) # 确保顺序一致
        ]
        return tuple(key_parts)

    # 1. 选择排名前 N 的配置
    print(f"选择策略 1: 添加排名前 {top_n} 的配置...")
    for config in configs[:top_n]:
        h = config_hash(config)
        if h not in selected_hashes:
            selected_configs.append(config)
            selected_hashes.add(h)
            print(f"  - 添加 Top {config['rank']}: {config['model_type']}, {config['structure_option']}, {config['activation_fn']}, {config['optimizer_type']}, Reg: {config['regularization_methods_str']} (Acc: {config['accuracy_10_percent']}%)")

    # 2. 确保包含一些顶尖的 MLP 配置
    print(f"选择策略 2: 添加排名前 {top_mlp} 的 MLP 配置 (如果不在列表中)...")
    mlp_count = 0
    for config in configs:
        if mlp_count >= top_mlp:
            break
        if config['model_type'] == 'mlp':
            h = config_hash(config)
            if h not in selected_hashes:
                selected_configs.append(config)
                selected_hashes.add(h)
                mlp_count += 1
                print(f"  - 添加 Top MLP (Rank {config['rank']}): {config['model_type']}, {config['structure_option']}, {config['activation_fn']}, {config['optimizer_type']}, Reg: {config['regularization_methods_str']} (Acc: {config['accuracy_10_percent']}%)")

    # 3. 确保包含一些顶尖的 SGD 配置
    print(f"选择策略 3: 添加排名前 {top_sgd} 的 SGD 配置 (如果不在列表中)...")
    sgd_count = 0
    for config in configs:
        if sgd_count >= top_sgd:
            break
        if config['optimizer_type'] == 'sgd':
            h = config_hash(config)
            if h not in selected_hashes:
                selected_configs.append(config)
                selected_hashes.add(h)
                sgd_count += 1
                print(f"  - 添加 Top SGD (Rank {config['rank']}): {config['model_type']}, {config['structure_option']}, {config['activation_fn']}, {config['optimizer_type']}, Reg: {config['regularization_methods_str']} (Acc: {config['accuracy_10_percent']}%)")

    # 4. 确保包含一些使用 Sigmoid/Tanh 的顶尖配置
    print(f"选择策略 4: 添加排名前 {top_other_activations} 的 Sigmoid/Tanh 配置 (如果不在列表中)...")
    other_act_count = 0
    for config in configs:
        if other_act_count >= top_other_activations:
            break
        if config['activation_fn'] in ['sigmoid', 'tanh']:
            h = config_hash(config)
            if h not in selected_hashes:
                selected_configs.append(config)
                selected_hashes.add(h)
                other_act_count += 1
                print(f"  - 添加 Top {config['activation_fn']} (Rank {config['rank']}): {config['model_type']}, {config['structure_option']}, {config['optimizer_type']}, Reg: {config['regularization_methods_str']} (Acc: {config['accuracy_10_percent']}%)")

    print(f"总共选择了 {len(selected_configs)} 种配置进行全数据集训练。")
    return selected_configs

# --- 全数据集训练逻辑 ---

def run_full_training(base_config, selected_config_info, full_epochs):
    """为单个选定的配置运行全数据集训练"""
    print("-" * 40)
    print(f"开始全数据集训练 (Rank {selected_config_info['rank']} based on 10% data)")
    print(f"配置: {selected_config_info['model_type']}, {selected_config_info['structure_option']}, {selected_config_info['activation_fn']}, {selected_config_info['optimizer_type']}, Loss: {selected_config_info['loss_type']}, Pool: {selected_config_info.get('pool_type', '-')}, Reg: {selected_config_info['regularization_methods_str']}")

    # 构建完整的配置字典
    config = base_config.copy()
    config.update({
        'model_type': selected_config_info['model_type'],
        'structure_option': selected_config_info['structure_option'],
        'activation_fn': selected_config_info['activation_fn'],
        'optimizer_type': selected_config_info['optimizer_type'],
        'loss_type': selected_config_info['loss_type'],
        'pool_type': selected_config_info.get('pool_type'), # 可能为 None
        'regularization_methods': selected_config_info.get('regularization_methods', []),
        # 使用固定/默认值或从 base_config 继承
        'learning_rate': config.get('learning_rate', 0.01),
        'momentum': config.get('momentum', 0.9), # SGD 需要
        'dropout_rate': config.get('dropout_rate', 0.3), # 如果正则化包含 dropout
        'patience': config.get('patience', 5), # 如果正则化包含 early_stopping
        'num_epochs': full_epochs, # 使用完整的训练轮数
        'use_augmented_data': False # 默认不使用增强，或者可以从md文件解析？当前未解析
    })
    # 确保use_one_hot根据loss_type设置
    config['use_one_hot'] = config['loss_type'] == 'mse'

    try:
        # 1. 准备全数据集加载器
        print("准备全数据集加载器...")
        # `prepare_data_loaders` 默认加载全部数据
        train_loader, test_loader = prepare_data_loaders(config)
        print(f"全训练集大小: {len(train_loader.dataset)}, 全测试集大小: {len(test_loader.dataset)}")

        # 2. 创建模型
        print("创建模型...")
        dropout_rate_actual = config['dropout_rate'] if 'dropout' in config.get('regularization_methods', []) else 0.0
        model = create_model(
            config['model_type'],
            config['structure_option'],
            dropout_rate_actual,
            config['activation_fn'],
            config.get('pool_type', 'max') # 提供默认值
        )

        # 3. 训练模型
        print(f"开始训练 {full_epochs} 轮...")
        start_time = time.time()
        # 注意: train_model 可能需要根据正则化配置中的 'early_stopping' 进行调整
        # 当前 train_model 似乎没有实现早停逻辑，如果需要，需修改 function.py
        model, history = train_model(model, train_loader, None, config, None) # 不用进度回调
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.2f} 秒")

        # 4. 测试模型
        print("测试模型...")
        accuracy = test_model(model, test_loader, config['device'])
        print(f"全数据集准确率: {accuracy:.2f}%")

        # 5. 清理GPU内存 (如果使用)
        if 'cuda' in config['device']:
            model = model.to('cpu') # 移到CPU
            del model # 删除模型引用
            clear_gpu_memory() # 清理缓存

        return {
            "full_accuracy": accuracy,
            "full_training_time": training_time,
            "error": None
        }

    except Exception as e:
        print(f"错误: 在为配置 Rank {selected_config_info['rank']} 进行全数据集训练时出错: {e}")
        import traceback
        print(traceback.format_exc())
        if 'cuda' in config.get('device', 'cpu'):
             clear_gpu_memory() # 尝试清理
        return {
            "full_accuracy": 0.0,
            "full_training_time": 0.0,
            "error": str(e)
        }
    finally:
         print("-" * 40)


# --- 保存最终结果 ---

def save_full_results(all_final_results):
    """将全数据集训练结果保存为Markdown和CSV"""
    if not all_final_results:
        print("没有可保存的最终结果。")
        return

    # 按全数据集准确率排序
    sorted_results = sorted(all_final_results, key=lambda x: x[1]['full_accuracy'], reverse=True)

    # --- 保存为 Markdown ---
    md_content = "# 全数据集训练 - 最终结果\n\n"
    md_content += f"对从10%数据测试中选出的 **{len(sorted_results)}** 种配置进行了全数据集训练。\n\n"
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    md_content += f"测试时间: {current_time}\n\n"
    md_content += "## 配置按最终准确率排序\n\n"
    md_content += "| 最终排名 | 10%数据排名 | 模型类型 | 结构 | 激活函数 | 优化器 | 损失 | 池化 | 正则化 | 最终准确率 | 训练时间(秒) | 错误 |\n"
    md_content += "|----------|-------------|----------|------|----------|--------|------|------|----------|------------|---------------|------|\n"

    data_for_csv = []

    for i, (config, result) in enumerate(sorted_results):
        pool_type = config.get('pool_type', '-') if config['model_type'] == 'cnn' else '-'
        reg_str = config['regularization_methods_str'] if config['regularization_methods_str'] else '无'
        error_msg = result['error'] if result['error'] else '-'

        md_content += (f"| {i+1} | {config['rank']} | {config['model_type'].upper()} | {config['structure_option']} | "
                       f"{config['activation_fn']} | {config['optimizer_type']} | {config['loss_type']} | "
                       f"{pool_type} | {reg_str} | {result['full_accuracy']:.2f}% | "
                       f"{result['full_training_time']:.2f} | {error_msg} |\n")

        # 为CSV准备数据
        row = config.copy() # 复制基础配置
        row['final_rank'] = i + 1
        row['initial_rank_10_percent'] = config['rank']
        row['full_accuracy'] = result['full_accuracy']
        row['full_training_time'] = result['full_training_time']
        row['error'] = result['error']
        # 清理一些辅助字段
        row.pop('rank', None)
        row.pop('accuracy_10_percent', None)
        row.pop('training_time_10_percent', None)
        row.pop('regularization_methods_str', None)
        row['regularization_methods'] = ','.join(row['regularization_methods']) if row['regularization_methods'] else '无' # 转为字符串

        data_for_csv.append(row)


    md_path = os.path.join(FULL_RESULTS_DIR, 'full_training_summary.md')
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"最终结果 Markdown 文件已保存到: {md_path}")
    except IOError as e:
        print(f"错误: 无法写入 Markdown 文件 {md_path}: {e}")

    # --- 保存为 CSV ---
    csv_path = os.path.join(FULL_RESULTS_DIR, 'full_training_results.csv')
    try:
        df = pd.DataFrame(data_for_csv)
        # 重新排列表头顺序，使关键信息靠前
        cols_order = ['final_rank', 'initial_rank_10_percent', 'full_accuracy', 'model_type',
                      'structure_option', 'activation_fn', 'optimizer_type', 'loss_type', 'pool_type',
                      'regularization_methods', 'full_training_time', 'error']
        # 添加可能缺失的列（例如旧md文件没有pool_type）
        for col in cols_order:
             if col not in df.columns:
                  df[col] = None
        df = df[cols_order]
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"最终结果 CSV 文件已保存到: {csv_path}")
    except IOError as e:
        print(f"错误: 无法写入 CSV 文件 {csv_path}: {e}")
    except Exception as e:
        print(f"错误: 创建或保存 CSV 时出错: {e}")


# --- 主函数 ---

def main():
    parser = argparse.ArgumentParser(description='根据10%数据测试结果选择配置进行全数据集训练')
    parser.add_argument('--md-file', type=str, default='配置测试结果/results_summary.md',
                        help='包含10%数据测试结果的Markdown文件路径')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU进行训练')
    parser.add_argument('--full-epochs', type=int, default=DEFAULT_FULL_EPOCHS,
                        help=f'全数据集训练的轮数 (默认: {DEFAULT_FULL_EPOCHS})')
    parser.add_argument('--top-n', type=int, default=20, help='选择10%测试中排名前N的配置')
    parser.add_argument('--top-mlp', type=int, default=5, help='额外选择排名前N的MLP配置')
    parser.add_argument('--top-sgd', type=int, default=2, help='额外选择排名前N的SGD配置')
    parser.add_argument('--top-other-act', type=int, default=2, help='额外选择排名前N的Sigmoid/Tanh配置')

    args = parser.parse_args()

    print("====== 开始全数据集训练选择与执行 ======")

    # 设置GPU (如果需要)
    fixed_config = {} # 先初始化
    if args.use_gpu:
        has_gpu, device = setup_gpu()
        if not has_gpu:
            print("GPU 不可用，将使用 CPU。")
            args.use_gpu = False
            fixed_config['device'] = 'cpu'
            fixed_config['batch_size'] = 32 # CPU batch size
        else:
            fixed_config['device'] = device
            fixed_config['batch_size'] = 64 # GPU batch size
    else:
        print("将使用 CPU 进行训练。")
        fixed_config['device'] = 'cpu'
        fixed_config['batch_size'] = 32

    # 添加其他固定参数
    fixed_config['learning_rate'] = 0.01 # 或者可以设为参数
    fixed_config['dropout_rate'] = 0.3
    fixed_config['patience'] = 5
    fixed_config['momentum'] = 0.9

    # 获取数据目录
    fixed_config['data_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'MNIST')
    if not os.path.exists(fixed_config['data_dir']):
         # 尝试上一级目录找dataset
         alt_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'MNIST')
         if os.path.exists(alt_data_dir):
              fixed_config['data_dir'] = alt_data_dir
         else:
              print(f"错误：无法找到 MNIST 数据集目录。尝试路径：{fixed_config['data_dir']} 和 {alt_data_dir}")
              sys.exit(1)
    print(f"使用数据目录: {fixed_config['data_dir']}")


    # 1. 解析结果文件
    parsed_configs = parse_markdown_results(args.md_file)
    if not parsed_configs:
        print("无法解析配置或文件为空，程序退出。")
        sys.exit(1)

    # 2. 选择配置
    selected_configs = select_configs_for_full_training(
        parsed_configs, args.top_n, args.top_mlp, args.top_sgd, args.top_other_act
    )
    if not selected_configs:
        print("未能选择任何配置进行训练，程序退出。")
        sys.exit(1)

    # 3. 运行全数据集训练
    final_results = []
    total_selected = len(selected_configs)
    print(f"\n====== 开始对选择的 {total_selected} 个配置进行全数据集训练 ======")
    for i, config_info in enumerate(selected_configs):
        print(f"\n--- 处理第 {i+1}/{total_selected} 个选定配置 ---")
        result = run_full_training(fixed_config, config_info, args.full_epochs)
        final_results.append((config_info, result)) # 存储原始信息和新结果

    # 4. 保存最终结果
    print("\n====== 所有选定配置训练完成 ======")
    save_full_results(final_results)

    print("脚本执行完毕。")

if __name__ == "__main__":
    main()
