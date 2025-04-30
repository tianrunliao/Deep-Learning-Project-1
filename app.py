from flask import Flask, render_template, request, jsonify
import threading
import time
import numpy as np
from function import create_model, prepare_data_loaders, train_model as real_train, test_model, get_batch

app = Flask(__name__)

# 存储训练任务状态
training_jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train_model_endpoint():
    """处理训练请求并启动真实的训练过程 (NumPy)"""
    # 获取前端发送的配置数据
    config = request.json
    
    # 检查模型类型 (只支持 MLP)
    if config.get('model_type', 'mlp').lower() != 'mlp':
        return jsonify({"success": False, "error": "当前后端仅支持 MLP 模型 (NumPy 版本)"}), 400
    
    # 生成唯一的任务ID
    job_id = str(int(time.time()))
    
    # 初始化任务状态
    training_jobs[job_id] = {
        "status": "pending", # 改为 pending，线程启动后变 running
        "progress": 0,
        "config": config # 存储原始前端配置
    }
    
    # 启动真实训练线程
    print(f"启动 NumPy 训练任务，Job ID: {job_id}")
    threading.Thread(target=real_training_numpy, args=(job_id,)).start()
    
    return jsonify({"success": True, "job_id": job_id})

@app.route('/api/train/status/<job_id>', methods=['GET'])
def get_training_status(job_id):
    if job_id not in training_jobs:
        return jsonify({"success": False, "error": "Job not found"}), 404
    
    job_data = training_jobs[job_id]
    # 安全地获取结果，避免在任务失败或未完成时出错
    response_data = {
        "status": job_data.get("status", "unknown"),
        "progress": job_data.get("progress", 0),
        "config": job_data.get("config", {}),
        "results": job_data.get("results"), # 可能为 None
        "error": job_data.get("error") # 可能为 None
    }
    return jsonify(response_data)

def real_training_numpy(job_id):
    """执行真实的 NumPy MLP 训练过程"""
    if job_id not in training_jobs:
        print(f"错误: 找不到 Job ID {job_id}")
        return
    
    job = training_jobs[job_id]
    job["status"] = "running"
    config = job["config"] # 前端传入的原始配置
    
    try:
        # 转换前端配置为 NumPy function.py 需要的格式
        training_config = {
            'model_type': 'mlp', # 固定为 mlp
            'structure_option': config.get('structure_option', 'medium'),
            'activation_fn': config.get('activation_fn', 'relu'),
            'loss_type': config.get('loss_type', 'cross_entropy'),
            'optimizer_type': config.get('optimizer_type', 'sgd'), # 只有 sgd
            'learning_rate': config.get('learning_rate', 0.1), # NumPy 学习率
            'momentum': config.get('momentum', 0.9), # 如果是 sgd
            'num_epochs': config.get('num_epochs', 10),
            'batch_size': 64, # 固定 batch size
            'use_augmented_data': False, # 固定为 False
            'device': 'cpu', # 固定为 cpu
            'data_dir': '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST', # 需要确保路径有效
            'use_one_hot': config.get('loss_type', 'cross_entropy') == 'mse'
        }
        # 验证数据目录是否存在
        if not os.path.exists(training_config['data_dir']):
            alt_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'MNIST')
            if os.path.exists(alt_data_dir):
                training_config['data_dir'] = alt_data_dir
            else:
                raise FileNotFoundError(f"MNIST 数据目录未找到: {training_config['data_dir']} 或 {alt_data_dir}")
        
        # 设置正则化配置 (直接从前端获取)
        reg_methods = config.get('regularization_methods', [])
        dropout_rate_config = config.get('dropout_rate', 0.3) # 从前端获取 dropout 率
        training_config['regularization_config'] = {
            'methods': reg_methods,
            'dropout_rate': dropout_rate_config if 'dropout' in reg_methods else 0.0,
            'early_stopping': False, # Web UI 暂不启用早停
            'patience': 5
        }
        # 将实际使用的 dropout_rate 存入主配置，以便模型创建使用
        training_config['dropout_rate'] = training_config['regularization_config']['dropout_rate']
        
        # 设置进度回调函数 (适配 NumPy 版 train_model)
        def progress_callback(epoch, total_epochs):
            # epoch 是从 0 开始的索引
            # 进度 = (当前完成的轮数 / 总轮数) * 100
            progress = int(((epoch + 1) / total_epochs) * 100)
            # 更新 job 状态
            job["progress"] = progress
            print(f"Job {job_id}: Epoch {epoch+1}/{total_epochs} complete, Progress: {progress}% ")
        
        # 准备数据加载器信息 (NumPy)
        print(f"Job {job_id}: 准备数据加载器信息...")
        train_loader_info, test_loader_info = prepare_data_loaders(training_config)
        
        # 创建模型 (NumPy)
        print(f"Job {job_id}: 创建 MLP 模型...")
        model = create_model(
            training_config['model_type'],
            training_config['structure_option'],
            training_config['dropout_rate'], # 使用配置中确定的 dropout 率
            training_config['activation_fn']
        )
        
        # 训练模型 (NumPy)
        print(f"Job {job_id}: 开始训练...")
        model, history = real_train(model, train_loader_info, None, training_config, progress_callback)
        
        # 训练完成后设置100% (确保回调最后设置为100%)
        job["progress"] = 100
        print(f"Job {job_id}: 训练完成.")
        
        # 测试模型 (NumPy)
        print(f"Job {job_id}: 开始测试...")
        accuracy = test_model(model, test_loader_info, training_config['device'], training_config['loss_type'])
        print(f"Job {job_id}: 测试完成, 准确率: {accuracy:.2f}%")
        
        # 保存结果
        job["results"] = {
            "accuracy": accuracy,
            "model_type": config["model_type"], # 使用前端传入的类型
            "mode": "真实训练 (NumPy)",
            "structure_html": generate_model_structure_html(config), # 使用前端配置生成HTML
            "config_summary": get_config_summary(config), # 使用前端配置生成摘要
        }
        
        job["status"] = "completed"
    
    except NotImplementedError as e:
        job["status"] = "failed"
        job["error"] = f"功能未实现: {str(e)}"
        print(f"Job {job_id} 训练失败 (NotImplementedError): {e}")
    except FileNotFoundError as e:
        job["status"] = "failed"
        job["error"] = f"数据文件错误: {str(e)}"
        print(f"Job {job_id} 训练失败 (FileNotFoundError): {e}")
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"训练过程中发生错误: {str(e)}"
        print(f"Job {job_id} 训练失败: {e}")
        import traceback
        print(traceback.format_exc())

def generate_model_structure_html(config):
    """生成模型结构的HTML表示 (只处理 MLP)"""
    model_type = config["model_type"]
    structure_option = config["structure_option"]
    activation_fn = config.get("activation_fn", "relu")
    
    if model_type == "mlp":
        # 获取MLP结构
        if structure_option == "small":
            layers = [784, 128, 10] # 输入层, 隐藏层, 输出层
        elif structure_option == "medium":
            layers = [784, 256, 128, 10]
        else: # large
            layers = [784, 512, 256, 128, 10]
        
        html = '<div class="model-structure">'
        html += '<h5>模型结构示意图</h5>'
        html += '<div class="layers-container">'
        
        for i, layer_size in enumerate(layers):
            # 为所有隐藏层使用相同的宽度（120px），只为输入层和输出层使用特殊宽度
            if i == 0:
                layer_width = 80  # 输入层宽度
            elif i == len(layers) - 1:
                layer_width = 80  # 输出层宽度
            else:
                layer_width = 120  # 所有隐藏层统一宽度
                
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
        
    else: # CNN
        # 由于后端只支持 MLP，理论上不会进入这里，但保留一个占位符
        return "<p>CNN 结构图不可用 (后端不支持)</p>"

def get_config_summary(config):
    """获取配置摘要，用于在结果中显示 (移除 CNN 相关)"""
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
        summary.append(f"Dropout 率: {config.get('dropout_rate', '未指定')}")
    
    return summary

if __name__ == '__main__':
    # 尝试查找并使用主机 IP 地址，如果需要从其他设备访问
    host_ip = '0.0.0.0' # 默认监听所有接口
    port = 5003 # 可以修改端口
    print(f"Flask 应用将在 http://{host_ip}:{port}/ 启动")
    # 增加 use_reloader=False 避免 Windows 下多线程问题和重复启动
    app.run(debug=True, host=host_ip, port=port, use_reloader=False)
