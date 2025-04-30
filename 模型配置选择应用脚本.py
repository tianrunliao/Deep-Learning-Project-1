# 必要导入
# import torch # 移除
import numpy as np # 确保导入 numpy
import pandas as pd
import os
import sys
import time
import argparse
import re # 用于解析Markdown

# 获取包含此脚本的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 确保脚本所在目录在Python搜索路径中
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# 尝试导入 NumPy 版本的 function 模块
try:
    from function import (
        create_model, prepare_data_loaders, train_model, test_model,
        # 移除了 GPU 相关导入: setup_gpu, print_gpu_memory_usage, clear_gpu_memory
        get_batch # 可能需要，虽然此脚本主要是调用高级函数
    )
    print("成功导入 NumPy 版 'function' 模块。")
except ImportError as e:
    print(f"错误：无法导入 function.py 模块 (NumPy 版)。")
    print(f"脚本目录: {script_dir}")
    print(f"Python 搜索路径 (sys.path): {sys.path}")
    print(f"原始错误: {e}")
    sys.exit(1)

# --- 配置和常量 ---
FULL_RESULTS_DIR = "完整数据集测试结果_NumPy" # 区分 NumPy 结果
os.makedirs(FULL_RESULTS_DIR, exist_ok=True)
DEFAULT_FULL_EPOCHS = 10 # 全数据集训练轮数

# --- 解析 Markdown 结果 ---

def parse_markdown_results(md_filepath):
    """解析 NumPy 版 results_summary_numpy.md 文件中的表格，返回配置列表"""
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

            # 寻找包含关键字的表头行 (NumPy 版表头没有池化)
            if line.startswith('|') and '排名' in line and '模型类型' in line and '正则化方法' in line and not header_found:
                header_line = line
                header_found = True
                print(f"找到潜在表头行: {header_line}")

            # 寻找分隔符行
            if header_found and line.startswith('|------|'):
                separator_index = i
                print(f"找到分隔符行于索引: {separator_index}")
                break

        if not header_line or separator_index == -1:
            print("错误：未能正确识别 Markdown 表格的表头或分隔符。")
            return []

        # 3. 解析表头 (NumPy 版表头)
        header = [h.strip() for h in header_line.strip('|').split('|') if h.strip()]
        required_cols = ['排名', '模型类型', '结构大小', '激活函数', '优化器', '损失函数', '正则化方法', '准确率']
        if not all(col in header for col in required_cols):
            print(f"错误：Markdown文件表头缺失必需列。需要: {required_cols}，找到：{header}")
            return []
        header_map = {h: i for i, h in enumerate(header)} # 创建列名到索引的映射
        print(f"成功解析表头: {header}")

        # 4. 解析数据行
        table_started = False
        for line in lines[separator_index + 1:]:
            line = line.strip()
            if not line or not line.startswith('|'): # 跳过空行和非表格行
                continue

            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) >= len(header):
                try:
                    config = {
                        'rank': int(parts[header_map['排名']]),
                        'model_type': parts[header_map['模型类型']].lower(),
                        'structure_option': parts[header_map['结构大小']].lower(),
                        'activation_fn': parts[header_map['激活函数']].lower(),
                        'optimizer_type': parts[header_map['优化器']].lower(),
                        'loss_type': parts[header_map['损失函数']].lower(),
                        # pool_type 不再存在于 NumPy 版的表头
                        'regularization_methods_str': parts[header_map['正则化方法']],
                        'accuracy_subset': float(parts[header_map['准确率']].rstrip('%')), # 改名为 accuracy_subset
                        'training_time_subset': float(parts[header_map.get('训练时间(秒)', -1)]) if '训练时间(秒)' in header_map and parts[header_map.get('训练时间(秒)')] != '-' else 0.0 # 改名
                    }

                    # 只处理 MLP 模型
                    if config['model_type'] != 'mlp':
                         print(f"警告: 跳过非 MLP 模型配置 (Rank {config['rank']})")
                         continue

                    # 解析正则化方法
                    reg_str = config['regularization_methods_str']
                    if reg_str == '无' or not reg_str or reg_str == '-':
                        config['regularization_methods'] = []
                    else:
                        config['regularization_methods'] = [m.strip() for m in reg_str.split(',') if m.strip()]

                    configs.append(config)
                except (ValueError, IndexError, KeyError) as e:
                    print(f"警告：解析数据行时出错: '{line}' - 错误: {e} - 列索引映射: {header_map}")
                    continue
            else:
                print(f"警告：跳过格式不匹配的数据行: '{line}' (期望至少 {len(header)} 列, 找到 {len(parts)}) ")

        print(f"成功从 Markdown 文件解析了 {len(configs)} 条 MLP 配置。")
        return configs

    except FileNotFoundError:
        print(f"错误: Markdown 结果文件未找到: {md_filepath}")
        return []
    except Exception as e:
        print(f"错误: 解析 Markdown 文件时发生未知错误: {e}")
        return []

# --- 配置选择逻辑 (基本不变，但只处理 MLP) ---

def select_configs_for_full_training(configs, top_n=20, top_sgd=2, top_other_activations=2):
    """根据预定义的策略选择 MLP 配置进行全数据集训练"""
    if not configs:
        return []

    # 确保只处理 MLP 配置
    mlp_configs = [cfg for cfg in configs if cfg['model_type'] == 'mlp']
    if not mlp_configs:
         print("未找到任何 MLP 配置可供选择。")
         return []

    selected_configs = []
    selected_hashes = set() # 用于去重

    def config_hash(cfg):
        key_parts = [
            cfg['model_type'], cfg['structure_option'], cfg['activation_fn'],
            cfg['optimizer_type'], cfg['loss_type'],
            # 移除 pool_type
            ','.join(sorted(cfg.get('regularization_methods', [])))
        ]
        return tuple(key_parts)

    # 1. 选择排名前 N 的 MLP 配置
    print(f"选择策略 1: 添加排名前 {top_n} 的 MLP 配置...")
    for config in mlp_configs[:top_n]:
        h = config_hash(config)
        if h not in selected_hashes:
            selected_configs.append(config)
            selected_hashes.add(h)
            print(f"  - 添加 Top MLP (Rank {config['rank']}): {config['structure_option']}, {config['activation_fn']}, {config['optimizer_type']}, Reg: {config['regularization_methods_str']} (Acc: {config['accuracy_subset']}%) ")

    # 2. 确保包含一些顶尖的 SGD 配置 (因为目前只有 SGD)
    # 这个策略在只有 SGD 时意义不大，但保留框架
    print(f"选择策略 2: 添加排名前 {top_sgd} 的 SGD 配置 (如果不在列表中)...")
    sgd_count = 0
    for config in mlp_configs:
        if sgd_count >= top_sgd:
            break
        if config['optimizer_type'] == 'sgd':
            h = config_hash(config)
            if h not in selected_hashes:
                selected_configs.append(config)
                selected_hashes.add(h)
                sgd_count += 1
                print(f"  - 添加 Top SGD (Rank {config['rank']}): {config['structure_option']}, {config['activation_fn']}, Reg: {config['regularization_methods_str']} (Acc: {config['accuracy_subset']}%) ")

    # 3. 确保包含一些使用 Sigmoid/Tanh 的顶尖配置
    print(f"选择策略 3: 添加排名前 {top_other_activations} 的 Sigmoid/Tanh 配置 (如果不在列表中)...")
    other_act_count = 0
    for config in mlp_configs:
        if other_act_count >= top_other_activations:
            break
        if config['activation_fn'] in ['sigmoid', 'tanh']:
            h = config_hash(config)
            if h not in selected_hashes:
                selected_configs.append(config)
                selected_hashes.add(h)
                other_act_count += 1
                print(f"  - 添加 Top {config['activation_fn']} (Rank {config['rank']}): {config['structure_option']}, {config['optimizer_type']}, Reg: {config['regularization_methods_str']} (Acc: {config['accuracy_subset']}%) ")

    print(f"总共选择了 {len(selected_configs)} 种 MLP 配置进行全数据集训练。")
    return selected_configs

# --- 全数据集训练逻辑 (NumPy 版本) ---

def run_full_training(base_config, selected_config_info, full_epochs):
    """为单个选定的配置运行全数据集训练 (NumPy)"""
    print("-" * 40)
    print(f"开始全数据集训练 (Rank {selected_config_info['rank']} based on subset data)")
    print(f"配置: MLP, {selected_config_info['structure_option']}, {selected_config_info['activation_fn']}, {selected_config_info['optimizer_type']}, Loss: {selected_config_info['loss_type']}, Reg: {selected_config_info['regularization_methods_str']}")

    # 构建完整的配置字典 (NumPy)
    config = base_config.copy() # 基础包含 device, batch_size, data_dir 等
    config.update({
        'model_type': 'mlp', # 固定为 MLP
        'structure_option': selected_config_info['structure_option'],
        'activation_fn': selected_config_info['activation_fn'],
        'optimizer_type': selected_config_info['optimizer_type'],
        'loss_type': selected_config_info['loss_type'],
        # pool_type 移除
        'regularization_methods': selected_config_info.get('regularization_methods', []),
        # 使用固定/默认值或从 base_config 继承
        'learning_rate': config.get('learning_rate', 0.1), # NumPy learning rate
        'momentum': config.get('momentum', 0.9), # SGD 需要
        'dropout_rate': config.get('dropout_rate', 0.3), # 如果正则化包含 dropout
        'patience': config.get('patience', 5), # 如果启用早停 (当前未启用)
        'num_epochs': full_epochs, # 使用完整的训练轮数
        'use_augmented_data': False # 固定为 False
    })
    # 确保use_one_hot根据loss_type设置
    config['use_one_hot'] = config['loss_type'] == 'mse'
    # 配置正则化信息
    config['regularization_config'] = {
         'methods': config['regularization_methods'],
         'dropout_rate': config['dropout_rate'] if 'dropout' in config['regularization_methods'] else 0.0,
         'early_stopping': False, # 全数据集训练通常不在这里启用早停
         'patience': config['patience']
    }

    try:
        # 1. 准备全数据集加载器信息 (NumPy)
        print("准备全数据集加载器信息...")
        train_loader_info, test_loader_info = prepare_data_loaders(config)
        print(f"全训练集大小: {train_loader_info['num_samples']}, 全测试集大小: {test_loader_info['num_samples']}")

        # 2. 创建模型 (NumPy)
        print("创建 MLP 模型...")
        dropout_rate_actual = config['regularization_config']['dropout_rate']
        model = create_model(
            config['model_type'],
            config['structure_option'],
            dropout_rate_actual,
            config['activation_fn']
            # pool_type 移除
        )

        # 3. 训练模型 (NumPy)
        print(f"开始训练 {full_epochs} 轮...")
        start_time = time.time()
        # NumPy train_model
        model, history = train_model(model, train_loader_info, None, config, None) # 不用进度回调, 无验证集
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.2f} 秒")

        # 4. 测试模型 (NumPy)
        print("测试模型...")
        accuracy = test_model(model, test_loader_info, config['device'], config['loss_type'])
        print(f"全数据集准确率: {accuracy:.2f}%")

        # 5. 移除 GPU 清理

        return {
            "full_accuracy": accuracy,
            "full_training_time": training_time,
            "error": None
        }

    except NotImplementedError as e:
         print(f"错误: 配置包含未实现的功能: {e}")
         return {
             "full_accuracy": 0.0,
             "full_training_time": 0.0,
             "error": f"NotImplementedError: {e}"
         }
    except Exception as e:
        print(f"错误: 在为配置 Rank {selected_config_info['rank']} 进行全数据集训练时出错: {e}")
        import traceback
        print(traceback.format_exc())
        # 移除 GPU 清理
        return {
            "full_accuracy": 0.0,
            "full_training_time": 0.0,
            "error": str(e)
        }
    finally:
         print("-" * 40)


# --- 保存最终结果 (适配 NumPy) ---

def save_full_results(all_final_results):
    """将全数据集训练结果保存为Markdown和CSV (NumPy)"""
    if not all_final_results:
        print("没有可保存的最终结果。")
        return

    # 按全数据集准确率排序 (过滤错误)
    valid_results = [(cfg, res) for cfg, res in all_final_results if res.get('error') is None]
    if not valid_results:
         print("没有有效的最终结果可保存。")
         return
    sorted_results = sorted(valid_results, key=lambda x: x[1]['full_accuracy'], reverse=True)

    # --- 保存为 Markdown --- (移除 pool_type 列)
    md_content = "# 全数据集训练 - 最终结果 (NumPy)\n\n"
    md_content += f"对从子集数据测试中选出的 **{len(all_final_results)}** 种 MLP 配置进行了全数据集训练 (成功 {len(valid_results)} 种)。\n\n"
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    md_content += f"测试时间: {current_time}\n\n"
    md_content += "## 配置按最终准确率排序\n\n"
    md_content += "| 最终排名 | 子集排名 | 模型类型 | 结构 | 激活函数 | 优化器 | 损失 | 正则化 | 最终准确率 | 训练时间(秒) | 错误 |\n"
    md_content += "|----------|----------|----------|------|----------|--------|------|----------|------------|---------------|------|\n"

    data_for_csv = []

    for i, (config, result) in enumerate(sorted_results):
        # pool_type 移除
        reg_str = config['regularization_methods_str'] if config['regularization_methods_str'] else '无'
        error_msg = '-' # 既然过滤了，这里应该是 '-'

        md_content += (f"| {i+1} | {config['rank']} | {config['model_type'].upper()} | {config['structure_option']} | "
                       f"{config['activation_fn']} | {config['optimizer_type']} | {config['loss_type']} | "
                       f"{reg_str} | {result['full_accuracy']:.2f}% | "
                       f"{result['full_training_time']:.2f} | {error_msg} |\n")

        # 为CSV准备数据
        row = config.copy() # 复制基础配置
        row['final_rank'] = i + 1
        row['initial_rank_subset'] = config['rank'] # 重命名
        row['full_accuracy'] = result['full_accuracy']
        row['full_training_time'] = result['full_training_time']
        row['error'] = None # 既然过滤了，这里是 None
        # 清理一些辅助字段
        row.pop('rank', None)
        row.pop('accuracy_subset', None)
        row.pop('training_time_subset', None)
        row.pop('regularization_methods_str', None)
        row.pop('pool_type', None) # 确保移除
        row['regularization_methods'] = ','.join(row['regularization_methods']) if row['regularization_methods'] else '无' # 转为字符串

        data_for_csv.append(row)

    # 添加失败的配置到 Markdown (可选)
    failed_results = [(cfg, res) for cfg, res in all_final_results if res.get('error') is not None]
    if failed_results:
         md_content += "\n## 失败的配置\n\n"
         for config, result in failed_results:
              cfg_desc = f"MLP, {config.get('structure_option','?')}, {config.get('activation_fn','?')}, {config.get('optimizer_type','?')}, Loss: {config.get('loss_type','?')}, Reg: {config.get('regularization_methods_str','?')}"
              md_content += f"- Rank (Subset): {config.get('rank', 'N/A')} - 配置: {cfg_desc}\n"
              md_content += f"  - 错误: {result['error']}\n"


    md_path = os.path.join(FULL_RESULTS_DIR, 'full_training_summary_numpy.md')
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"最终结果 Markdown 文件已保存到: {md_path}")
    except IOError as e:
        print(f"错误: 无法写入 Markdown 文件 {md_path}: {e}")

    # --- 保存为 CSV --- (移除 pool_type)
    csv_path = os.path.join(FULL_RESULTS_DIR, 'full_training_results_numpy.csv')
    try:
        df = pd.DataFrame(data_for_csv)
        # 重新排列表头顺序
        cols_order = ['final_rank', 'initial_rank_subset', 'full_accuracy', 'model_type',
                      'structure_option', 'activation_fn', 'optimizer_type', 'loss_type',
                      'regularization_methods', 'full_training_time', 'error']
        # 添加可能缺失的列
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


# --- 主函数 (NumPy 版本) ---

def main():
    parser = argparse.ArgumentParser(description='根据子集数据测试结果选择 MLP 配置进行全数据集训练 (NumPy)')
    parser.add_argument('--md-file', type=str, default='配置测试结果_NumPy/results_summary_numpy.md',
                        help='包含子集数据 MLP 测试结果的Markdown文件路径')
    # parser.add_argument('--use-gpu', action='store_true', help='使用GPU进行训练') # 移除
    parser.add_argument('--full-epochs', type=int, default=DEFAULT_FULL_EPOCHS,
                        help=f'全数据集训练的轮数 (默认: {DEFAULT_FULL_EPOCHS})')
    parser.add_argument('--top-n', type=int, default=10, help='选择子集测试中排名前N的配置 (减少默认值)')
    # top-mlp 移除，因为只处理 MLP
    parser.add_argument('--top-sgd', type=int, default=1, help='额外选择排名前N的SGD配置 (减少默认值)')
    parser.add_argument('--top-other-act', type=int, default=1, help='额外选择排名前N的Sigmoid/Tanh配置 (减少默认值)')

    args = parser.parse_args()

    print("====== 开始全数据集训练选择与执行 (NumPy) ======")

    # 固定配置 (CPU)
    fixed_config = {}
    print("将使用 CPU 进行训练。")
    fixed_config['device'] = 'cpu'
    fixed_config['batch_size'] = 64

    # 添加其他固定参数
    fixed_config['learning_rate'] = 0.1 # NumPy 默认学习率
    fixed_config['dropout_rate'] = 0.3 # 默认 dropout (如果启用)
    fixed_config['patience'] = 5    # 早停耐心 (当前未用)
    fixed_config['momentum'] = 0.9    # 默认动量

    # 获取数据目录
    # (与遍历脚本相同的逻辑)
    script_dir_main = os.path.dirname(os.path.abspath(__file__))
    data_dir_local = os.path.join(script_dir_main, 'dataset', 'MNIST')
    data_dir_parent = os.path.join(os.path.dirname(script_dir_main), 'dataset', 'MNIST')

    if os.path.exists(data_dir_local):
         fixed_config['data_dir'] = data_dir_local
    elif os.path.exists(data_dir_parent):
         fixed_config['data_dir'] = data_dir_parent
    else:
         print(f"错误：无法找到 MNIST 数据集目录。尝试路径：{data_dir_local} 和 {data_dir_parent}")
         sys.exit(1)
    print(f"使用数据目录: {fixed_config['data_dir']}")


    # 1. 解析结果文件
    print(f"解析 Markdown 文件: {args.md_file}")
    parsed_configs = parse_markdown_results(args.md_file)
    if not parsed_configs:
        print("无法解析配置或文件为空/无有效配置，程序退出。")
        sys.exit(1)

    # 2. 选择配置
    selected_configs = select_configs_for_full_training(
        parsed_configs, args.top_n, args.top_sgd, args.top_other_act
    )
    if not selected_configs:
        print("未能选择任何配置进行训练，程序退出。")
        sys.exit(1)

    # 3. 运行全数据集训练
    final_results = []
    total_selected = len(selected_configs)
    print(f"\n====== 开始对选择的 {total_selected} 个配置进行全数据集训练 ({args.full_epochs} 轮) ======")
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
