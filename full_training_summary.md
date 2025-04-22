# 全数据集训练 - 最终结果

对从10%数据测试中选出的 **29** 种配置进行了全数据集训练。

测试时间: 2025-04-21 02:57:32

## 配置按最终准确率排序

| 最终排名 | 10%数据排名 | 模型类型 | 结构 | 激活函数 | 优化器 | 损失 | 池化 | 正则化 | 最终准确率 | 训练时间(秒) | 错误 |
|----------|-------------|----------|------|----------|--------|------|------|----------|------------|---------------|------|
| 1 | 24 | CNN | large | leaky_relu | sgd | cross_entropy | max | dropout, early_stopping | 99.26% | 474.01 | - |
| 2 | 25 | CNN | large | tanh | sgd | cross_entropy | max | dropout | 99.16% | 581.19 | - |
| 3 | 20 | CNN | large | leaky_relu | sgd | cross_entropy | max | 无 | 99.12% | 474.16 | - |
| 4 | 33 | CNN | large | tanh | sgd | cross_entropy | max | early_stopping | 99.12% | 524.99 | - |
| 5 | 7 | CNN | small | leaky_relu | adam | mse | avg | early_stopping | 99.01% | 155.88 | - |
| 6 | 2 | CNN | small | leaky_relu | adam | mse | max | 无 | 99.00% | 141.04 | - |
| 7 | 54 | CNN | large | tanh | sgd | cross_entropy | max | dropout, early_stopping | 98.99% | 526.25 | - |
| 8 | 19 | CNN | large | relu | sgd | cross_entropy | max | dropout, early_stopping | 98.88% | 509.40 | - |
| 9 | 4 | CNN | small | leaky_relu | adam | mse | avg | 无 | 98.84% | 160.64 | - |
| 10 | 3 | CNN | small | leaky_relu | adam | mse | avg | dropout, early_stopping | 98.74% | 146.69 | - |
| 11 | 14 | CNN | small | leaky_relu | adam | mse | avg | dropout | 98.61% | 184.53 | - |
| 12 | 1 | CNN | small | leaky_relu | adam | mse | max | early_stopping | 98.50% | 139.81 | - |
| 13 | 12 | CNN | small | leaky_relu | adam | cross_entropy | avg | 无 | 98.44% | 166.10 | - |
| 14 | 6 | CNN | small | leaky_relu | adam | mse | max | dropout, early_stopping | 98.40% | 139.61 | - |
| 15 | 10 | CNN | small | leaky_relu | adam | mse | max | dropout | 98.26% | 174.58 | - |
| 16 | 9 | CNN | medium | leaky_relu | adam | mse | avg | dropout, early_stopping | 98.18% | 315.60 | - |
| 17 | 18 | CNN | large | leaky_relu | adam | cross_entropy | avg | dropout | 97.55% | 484.19 | - |
| 18 | 15 | CNN | medium | relu | adam | mse | avg | early_stopping | 97.37% | 299.85 | - |
| 19 | 104 | MLP | medium | sigmoid | adam | cross_entropy | - | dropout, early_stopping | 97.26% | 21.42 | - |
| 20 | 107 | MLP | medium | sigmoid | adam | cross_entropy | - | dropout | 97.14% | 25.44 | - |
| 21 | 106 | MLP | small | sigmoid | adam | cross_entropy | - | early_stopping | 96.93% | 11.82 | - |
| 22 | 101 | MLP | small | sigmoid | adam | cross_entropy | - | 无 | 96.52% | 12.73 | - |
| 23 | 105 | MLP | medium | leaky_relu | adam | mse | - | 无 | 96.52% | 19.60 | - |
| 24 | 17 | CNN | medium | leaky_relu | adam | cross_entropy | avg | 无 | 96.51% | 288.82 | - |
| 25 | 16 | CNN | medium | leaky_relu | adam | mse | max | early_stopping | 95.28% | 294.11 | - |
| 26 | 11 | CNN | medium | leaky_relu | adam | mse | max | dropout | 89.25% | 320.25 | - |
| 27 | 8 | CNN | medium | leaky_relu | adam | mse | max | 无 | 55.25% | 320.28 | - |
| 28 | 5 | CNN | medium | leaky_relu | adam | mse | avg | early_stopping | 53.09% | 249.88 | - |
| 29 | 13 | CNN | large | relu | adam | cross_entropy | avg | 无 | 11.35% | 563.53 | - |
