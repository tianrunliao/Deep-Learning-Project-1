from flask import Flask, render_template, request, jsonify
import threading
import time
import numpy as np
from function import create_model, prepare_data_loaders, train_model as real_train, test_model
import torch

app = Flask(__name__)

# 存储训练任务状态
training_jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    """处理训练请求并启动真实的训练过程"""
    # 获取前端发送的配置数据
    config = request.json
    
    # 生成唯一的任务ID
    job_id = str(int(time.time()))
    
    # 初始化任务状态
    training_jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "config": config
    }
    
    # 启动真实训练线程
    threading.Thread(target=real_training, args=(job_id,)).start()
    
    return jsonify({"success": True, "job_id": job_id})

@app.route('/api/train/status/<job_id>', methods=['GET'])
def get_training_status(job_id):
    if job_id not in training_jobs:
        return jsonify({"success": False, "error": "Job not found"}), 404
    
    return jsonify(training_jobs[job_id])

def real_training(job_id):
    """执行真实的神经网络训练过程"""
    job = training_jobs[job_id]
    config = job["config"]
    
    try:
        # 转换前端配置为function.py需要的格式
        training_config = {
            'model_type': config['model_type'],
            'structure_option': config['structure_option'],
            'activation_fn': config.get('activation_fn', 'relu'),
            'pool_type': config.get('pool_type', 'max'),
            'loss_type': config.get('loss_type', 'cross_entropy'),
            'optimizer_type': config.get('optimizer_type', 'sgd'),
            'learning_rate': config.get('learning_rate', 0.01),
            'momentum': config.get('momentum', 0.9),
            'num_epochs': config.get('num_epochs', 10),
            'batch_size': 64,
            'use_augmented_data': config.get('use_augmented_data', False),
            'device': 'cpu',
            'data_dir': '/Users/tianrunliao/Desktop/廖天润 22300680285 project 1/dataset/MNIST',
            'use_one_hot': config.get('loss_type', 'cross_entropy') == 'mse'
        }
        
        # 设置正则化配置
        training_config['regularization_config'] = {
            'methods': config.get('regularization_methods', []),
            'dropout_rate': 0.3 if 'dropout' in config.get('regularization_methods', []) else 0.0,
            'l1_lambda': 0.0001,
            'weight_decay': 0.0001 if 'l2' in config.get('regularization_methods', []) else 0.0,
            'patience': 5
        }
        
        # 如果使用增强数据，调整学习率
        if config.get('use_augmented_data', False):
            training_config['learning_rate'] *= 0.5  # 降低学习率
            training_config['num_epochs'] += 5  # 增加训练轮次
        
        # 设置进度回调函数
        def progress_callback(epoch, total_epochs):
            progress = int((epoch + 1) / total_epochs * 100)
            # 确保进度是10的倍数或100%
            progress = (progress // 10) * 10
            if progress == 100 and epoch < total_epochs - 1:
                progress = 90  # 除非是最后一个epoch，否则最高显示90%
            job["progress"] = progress
        
        # 准备数据加载器
        train_loader, test_loader = prepare_data_loaders(training_config)
        
        # 创建模型
        model = create_model(
            training_config['model_type'],
            training_config['structure_option'],
            training_config['regularization_config'].get('dropout_rate', 0.0),
            training_config['activation_fn'],
            training_config.get('pool_type', 'max')
        )
        
        # 训练模型
        model, history = real_train(model, train_loader, None, training_config, progress_callback)
        
        # 训练完成后设置100%
        job["progress"] = 100
        
        # 测试模型
        accuracy = test_model(model, test_loader, training_config['device'])
        
        # 保存结果
        job["results"] = {
            "accuracy": accuracy,
            "model_type": config["model_type"],
            "mode": "真实训练",
            "structure_html": generate_model_structure_html(config),
            "config_summary": get_config_summary(config),
        }
        
        job["status"] = "completed"
    
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        print(f"训练出错: {e}")

def generate_model_structure_html(config):
    """生成模型结构的HTML表示"""
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
        html = '<div class="model-structure">'
        html += '<h5>CNN模型结构示意图</h5>'
        html += '<div class="cnn-container">'
        
        # 输入层
        html += '<div class="cnn-layer input-layer"><div class="layer-label">输入<br>28x28x1</div></div>'
        
        # 第一个卷积层
        conv1_filters = {'small': 16, 'medium': 32, 'large': 64}[structure_option]
        html += f'<div class="cnn-layer conv-layer"><div class="layer-label">卷积层 1<br>{conv1_filters}个3x3滤波器</div></div>'
        html += f'<div class="activation">{activation_fn}</div>'
        
        # 第一个池化层
        pool_type = config.get("pool_type", "max")
        html += f'<div class="cnn-layer pool-layer"><div class="layer-label">{pool_type}池化<br>14x14x{conv1_filters}</div></div>'
        
        # 第二个卷积层
        conv2_filters = {'small': 32, 'medium': 64, 'large': 128}[structure_option]
        html += f'<div class="cnn-layer conv-layer"><div class="layer-label">卷积层 2<br>{conv2_filters}个3x3滤波器</div></div>'
        html += f'<div class="activation">{activation_fn}</div>'
        
        # 第二个池化层
        html += f'<div class="cnn-layer pool-layer"><div class="layer-label">{pool_type}池化<br>7x7x{conv2_filters}</div></div>'
        
        # 全连接层
        html += f'<div class="cnn-layer fc-layer"><div class="layer-label">全连接层<br>128</div></div>'
        html += f'<div class="activation">{activation_fn}</div>'
        
        # 输出层
        html += '<div class="cnn-layer output-layer"><div class="layer-label">输出层<br>10</div></div>'
        
        html += '</div></div>'
        return html

def get_config_summary(config):
    """获取配置摘要，用于在结果中显示"""
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
    
    return summary

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
