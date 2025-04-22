# 神经网络模型配置测试结果

共测试了 **576** 种不同的配置组合，使用10%的训练数据进行快速评估。

测试时间: 2025-04-20 21:42:47

## 所有配置按准确率排序

| 排名 | 模型类型 | 结构大小 | 激活函数 | 优化器 | 损失函数 | 池化类型 | 正则化方法 | 准确率 | 训练时间(秒) |
|------|----------|----------|------------|--------|----------|----------|------------|---------|---------------|
| 1 | CNN | small | leaky_relu | adam | mse | max | early_stopping | 97.97% | 4.75 |
| 2 | CNN | small | leaky_relu | adam | mse | max | 无 | 97.92% | 4.52 |
| 3 | CNN | small | leaky_relu | adam | mse | avg | dropout, early_stopping | 97.79% | 4.64 |
| 4 | CNN | small | leaky_relu | adam | mse | avg | 无 | 97.78% | 4.28 |
| 5 | CNN | medium | leaky_relu | adam | mse | avg | early_stopping | 97.76% | 6.81 |
| 6 | CNN | small | leaky_relu | adam | mse | max | dropout, early_stopping | 97.73% | 6.50 |
| 7 | CNN | small | leaky_relu | adam | mse | avg | early_stopping | 97.68% | 4.56 |
| 8 | CNN | medium | leaky_relu | adam | mse | max | 无 | 97.60% | 6.96 |
| 9 | CNN | medium | leaky_relu | adam | mse | avg | dropout, early_stopping | 97.36% | 6.99 |
| 10 | CNN | small | leaky_relu | adam | mse | max | dropout | 97.29% | 4.81 |
| 11 | CNN | medium | leaky_relu | adam | mse | max | dropout | 97.11% | 6.91 |
| 12 | CNN | small | leaky_relu | adam | cross_entropy | avg | 无 | 97.06% | 4.50 |
| 13 | CNN | large | relu | adam | cross_entropy | avg | 无 | 97.06% | 17.07 |
| 14 | CNN | small | leaky_relu | adam | mse | avg | dropout | 97.02% | 4.63 |
| 15 | CNN | medium | relu | adam | mse | avg | early_stopping | 96.98% | 9.01 |
| 16 | CNN | medium | leaky_relu | adam | mse | max | early_stopping | 96.98% | 6.85 |
| 17 | CNN | medium | leaky_relu | adam | cross_entropy | avg | 无 | 96.93% | 6.75 |
| 18 | CNN | large | leaky_relu | adam | cross_entropy | avg | dropout | 96.93% | 14.45 |
| 19 | CNN | large | relu | sgd | cross_entropy | max | dropout, early_stopping | 96.86% | 13.13 |
| 20 | CNN | large | leaky_relu | sgd | cross_entropy | max | 无 | 96.80% | 12.20 |
| 21 | CNN | medium | leaky_relu | adam | cross_entropy | avg | early_stopping | 96.79% | 6.80 |
| 22 | CNN | medium | leaky_relu | adam | cross_entropy | max | 无 | 96.77% | 6.72 |
| 23 | CNN | small | leaky_relu | adam | cross_entropy | avg | early_stopping | 96.74% | 4.51 |
| 24 | CNN | large | leaky_relu | sgd | cross_entropy | max | dropout, early_stopping | 96.67% | 11.76 |
| 25 | CNN | large | tanh | sgd | cross_entropy | max | dropout | 96.61% | 13.05 |
| 26 | CNN | large | leaky_relu | adam | cross_entropy | avg | early_stopping | 96.58% | 14.53 |
| 27 | CNN | large | leaky_relu | sgd | cross_entropy | max | dropout | 96.55% | 12.36 |
| 28 | CNN | small | leaky_relu | adam | cross_entropy | max | dropout | 96.54% | 4.70 |
| 29 | CNN | small | leaky_relu | adam | cross_entropy | avg | dropout | 96.53% | 4.54 |
| 30 | CNN | large | relu | sgd | cross_entropy | max | 无 | 96.51% | 12.26 |
| 31 | CNN | small | leaky_relu | adam | cross_entropy | max | early_stopping | 96.50% | 4.54 |
| 32 | CNN | medium | relu | sgd | cross_entropy | max | early_stopping | 96.48% | 8.47 |
| 33 | CNN | large | tanh | sgd | cross_entropy | max | early_stopping | 96.48% | 13.60 |
| 34 | CNN | small | relu | adam | cross_entropy | avg | dropout | 96.47% | 4.20 |
| 35 | CNN | small | leaky_relu | adam | cross_entropy | max | dropout, early_stopping | 96.47% | 4.68 |
| 36 | CNN | medium | leaky_relu | adam | mse | avg | 无 | 96.45% | 6.69 |
| 37 | CNN | small | relu | adam | cross_entropy | avg | 无 | 96.43% | 4.17 |
| 38 | CNN | medium | leaky_relu | adam | cross_entropy | max | early_stopping | 96.41% | 6.81 |
| 39 | CNN | large | relu | adam | cross_entropy | avg | dropout | 96.31% | 14.53 |
| 40 | CNN | small | relu | sgd | cross_entropy | max | dropout | 96.29% | 4.00 |
| 41 | CNN | medium | leaky_relu | adam | cross_entropy | avg | dropout | 96.29% | 6.81 |
| 42 | CNN | large | leaky_relu | sgd | cross_entropy | max | early_stopping | 96.24% | 11.70 |
| 43 | CNN | large | relu | sgd | cross_entropy | max | early_stopping | 96.22% | 12.34 |
| 44 | CNN | small | leaky_relu | adam | cross_entropy | max | 无 | 96.20% | 4.76 |
| 45 | CNN | large | leaky_relu | adam | cross_entropy | max | dropout | 96.20% | 14.47 |
| 46 | CNN | medium | leaky_relu | sgd | cross_entropy | max | 无 | 96.19% | 10.41 |
| 47 | CNN | medium | leaky_relu | sgd | cross_entropy | max | dropout | 96.11% | 8.74 |
| 48 | CNN | small | relu | adam | cross_entropy | max | dropout, early_stopping | 96.09% | 4.55 |
| 49 | CNN | small | leaky_relu | adam | cross_entropy | avg | dropout, early_stopping | 96.07% | 4.51 |
| 50 | CNN | medium | relu | sgd | cross_entropy | max | 无 | 96.01% | 8.74 |
| 51 | CNN | medium | relu | sgd | cross_entropy | max | dropout | 95.97% | 8.34 |
| 52 | CNN | medium | relu | sgd | cross_entropy | max | dropout, early_stopping | 95.95% | 8.64 |
| 53 | CNN | medium | leaky_relu | adam | cross_entropy | max | dropout, early_stopping | 95.92% | 6.67 |
| 54 | CNN | large | tanh | sgd | cross_entropy | max | dropout, early_stopping | 95.92% | 12.99 |
| 55 | CNN | medium | tanh | sgd | cross_entropy | max | early_stopping | 95.83% | 8.89 |
| 56 | CNN | large | tanh | sgd | cross_entropy | max | 无 | 95.79% | 12.83 |
| 57 | CNN | large | relu | sgd | cross_entropy | max | dropout | 95.73% | 13.28 |
| 58 | CNN | small | leaky_relu | sgd | cross_entropy | max | dropout, early_stopping | 95.72% | 4.42 |
| 59 | CNN | large | leaky_relu | adam | cross_entropy | max | 无 | 95.71% | 12.02 |
| 60 | CNN | small | relu | adam | cross_entropy | avg | dropout, early_stopping | 95.68% | 4.20 |
| 61 | CNN | small | relu | sgd | cross_entropy | max | 无 | 95.66% | 3.98 |
| 62 | CNN | medium | tanh | sgd | cross_entropy | max | dropout, early_stopping | 95.65% | 9.16 |
| 63 | CNN | medium | leaky_relu | sgd | cross_entropy | max | dropout, early_stopping | 95.59% | 6.52 |
| 64 | CNN | small | leaky_relu | sgd | cross_entropy | max | dropout | 95.57% | 4.71 |
| 65 | CNN | medium | leaky_relu | adam | mse | max | dropout, early_stopping | 95.56% | 6.90 |
| 66 | CNN | medium | tanh | sgd | cross_entropy | max | dropout | 95.52% | 10.27 |
| 67 | CNN | small | leaky_relu | sgd | cross_entropy | max | early_stopping | 95.48% | 4.34 |
| 68 | CNN | small | relu | adam | cross_entropy | max | dropout | 95.45% | 4.22 |
| 69 | CNN | medium | relu | adam | cross_entropy | max | early_stopping | 95.44% | 8.78 |
| 70 | CNN | medium | leaky_relu | adam | cross_entropy | avg | dropout, early_stopping | 95.43% | 6.97 |
| 71 | CNN | large | leaky_relu | adam | cross_entropy | max | early_stopping | 95.40% | 15.62 |
| 72 | CNN | medium | leaky_relu | sgd | cross_entropy | max | early_stopping | 95.36% | 8.78 |
| 73 | CNN | medium | relu | adam | cross_entropy | avg | dropout | 95.35% | 9.97 |
| 74 | CNN | small | tanh | sgd | cross_entropy | max | dropout | 95.33% | 5.00 |
| 75 | CNN | large | leaky_relu | adam | cross_entropy | avg | 无 | 95.20% | 11.93 |
| 76 | CNN | small | relu | adam | cross_entropy | max | 无 | 95.13% | 4.16 |
| 77 | CNN | medium | relu | adam | mse | max | dropout | 95.12% | 8.80 |
| 78 | CNN | small | tanh | sgd | cross_entropy | max | early_stopping | 95.08% | 5.02 |
| 79 | CNN | small | relu | sgd | cross_entropy | max | early_stopping | 95.05% | 4.03 |
| 80 | CNN | medium | tanh | sgd | cross_entropy | max | 无 | 94.94% | 9.41 |
| 81 | CNN | small | tanh | sgd | cross_entropy | max | 无 | 94.90% | 4.94 |
| 82 | CNN | large | relu | adam | cross_entropy | max | early_stopping | 94.89% | 14.69 |
| 83 | CNN | small | leaky_relu | sgd | cross_entropy | max | 无 | 94.88% | 4.45 |
| 84 | CNN | medium | relu | adam | cross_entropy | avg | early_stopping | 94.77% | 8.63 |
| 85 | CNN | large | leaky_relu | adam | cross_entropy | avg | dropout, early_stopping | 94.70% | 14.07 |
| 86 | CNN | small | tanh | sgd | cross_entropy | max | dropout, early_stopping | 94.69% | 4.65 |
| 87 | CNN | small | relu | sgd | cross_entropy | max | dropout, early_stopping | 94.63% | 4.05 |
| 88 | CNN | medium | leaky_relu | adam | cross_entropy | max | dropout | 94.57% | 6.82 |
| 89 | CNN | large | relu | adam | cross_entropy | max | dropout | 94.53% | 13.97 |
| 90 | CNN | small | tanh | adam | mse | max | dropout, early_stopping | 94.50% | 5.31 |
| 91 | CNN | small | tanh | adam | cross_entropy | max | early_stopping | 94.36% | 4.87 |
| 92 | CNN | small | relu | adam | cross_entropy | avg | early_stopping | 94.24% | 4.12 |
| 93 | CNN | large | relu | adam | cross_entropy | avg | dropout, early_stopping | 94.14% | 19.10 |
| 94 | CNN | large | leaky_relu | adam | mse | avg | dropout | 94.09% | 14.64 |
| 95 | CNN | large | relu | adam | cross_entropy | avg | early_stopping | 94.06% | 13.98 |
| 96 | CNN | large | relu | sgd | cross_entropy | avg | early_stopping | 93.94% | 12.00 |
| 97 | CNN | large | leaky_relu | adam | mse | max | early_stopping | 93.94% | 14.36 |
| 98 | CNN | medium | relu | adam | cross_entropy | max | dropout | 93.79% | 8.75 |
| 99 | CNN | large | relu | sgd | cross_entropy | avg | dropout | 93.73% | 13.22 |
| 100 | CNN | large | leaky_relu | adam | mse | max | 无 | 93.66% | 14.44 |
| 101 | MLP | small | sigmoid | adam | cross_entropy | - | 无 | 93.65% | 0.38 |
| 102 | CNN | large | leaky_relu | sgd | cross_entropy | avg | dropout | 93.53% | 11.76 |
| 103 | CNN | small | tanh | adam | cross_entropy | max | 无 | 93.51% | 4.87 |
| 104 | MLP | medium | sigmoid | adam | cross_entropy | - | dropout, early_stopping | 93.49% | 0.62 |
| 105 | MLP | medium | leaky_relu | adam | mse | - | 无 | 93.30% | 0.53 |
| 106 | MLP | small | sigmoid | adam | cross_entropy | - | early_stopping | 93.27% | 0.33 |
| 107 | MLP | medium | sigmoid | adam | cross_entropy | - | dropout | 93.23% | 0.61 |
| 108 | MLP | medium | sigmoid | adam | mse | - | 无 | 93.20% | 0.52 |
| 109 | CNN | medium | relu | sgd | cross_entropy | avg | early_stopping | 93.19% | 8.69 |
| 110 | CNN | large | tanh | sgd | cross_entropy | avg | early_stopping | 93.04% | 13.64 |
| 111 | MLP | small | relu | adam | cross_entropy | - | early_stopping | 92.95% | 0.42 |
| 112 | MLP | small | sigmoid | adam | cross_entropy | - | dropout | 92.94% | 0.36 |
| 113 | MLP | small | sigmoid | adam | cross_entropy | - | dropout, early_stopping | 92.92% | 0.38 |
| 114 | MLP | small | tanh | adam | mse | - | 无 | 92.92% | 0.38 |
| 115 | CNN | large | leaky_relu | adam | mse | max | dropout | 92.92% | 14.61 |
| 116 | CNN | large | leaky_relu | sgd | cross_entropy | avg | 无 | 92.88% | 16.48 |
| 117 | CNN | medium | leaky_relu | sgd | cross_entropy | avg | dropout, early_stopping | 92.86% | 6.49 |
| 118 | CNN | large | tanh | sgd | cross_entropy | avg | 无 | 92.83% | 12.64 |
| 119 | MLP | medium | sigmoid | adam | mse | - | early_stopping | 92.78% | 0.54 |
| 120 | MLP | small | sigmoid | adam | mse | - | early_stopping | 92.73% | 0.32 |
| 121 | CNN | medium | relu | sgd | cross_entropy | avg | dropout | 92.72% | 8.37 |
| 122 | MLP | small | leaky_relu | adam | cross_entropy | - | dropout | 92.70% | 0.36 |
| 123 | CNN | large | tanh | sgd | cross_entropy | avg | dropout, early_stopping | 92.69% | 13.53 |
| 124 | MLP | small | relu | adam | cross_entropy | - | 无 | 92.67% | 0.35 |
| 125 | CNN | medium | relu | sgd | cross_entropy | avg | dropout, early_stopping | 92.66% | 8.45 |
| 126 | MLP | small | sigmoid | adam | mse | - | dropout, early_stopping | 92.63% | 0.35 |
| 127 | CNN | large | leaky_relu | adam | mse | avg | early_stopping | 92.63% | 14.43 |
| 128 | CNN | small | relu | sgd | cross_entropy | avg | 无 | 92.61% | 4.00 |
| 129 | CNN | medium | tanh | sgd | cross_entropy | avg | 无 | 92.61% | 8.97 |
| 130 | MLP | large | relu | adam | cross_entropy | - | 无 | 92.59% | 1.12 |
| 131 | MLP | large | leaky_relu | adam | mse | - | 无 | 92.59% | 1.09 |
| 132 | MLP | small | tanh | adam | cross_entropy | - | early_stopping | 92.58% | 0.34 |
| 133 | CNN | large | tanh | sgd | cross_entropy | avg | dropout | 92.57% | 13.86 |
| 134 | CNN | large | leaky_relu | adam | mse | max | dropout, early_stopping | 92.55% | 15.07 |
| 135 | CNN | small | leaky_relu | sgd | cross_entropy | avg | dropout | 92.53% | 4.19 |
| 136 | MLP | small | leaky_relu | adam | cross_entropy | - | early_stopping | 92.52% | 0.34 |
| 137 | MLP | small | leaky_relu | adam | mse | - | 无 | 92.52% | 0.38 |
| 138 | CNN | medium | leaky_relu | sgd | cross_entropy | avg | dropout | 92.50% | 8.80 |
| 139 | MLP | large | relu | adam | cross_entropy | - | early_stopping | 92.49% | 1.09 |
| 140 | MLP | small | leaky_relu | adam | cross_entropy | - | 无 | 92.48% | 0.36 |
| 141 | CNN | large | leaky_relu | sgd | cross_entropy | avg | early_stopping | 92.40% | 11.52 |
| 142 | MLP | large | sigmoid | adam | mse | - | early_stopping | 92.38% | 1.07 |
| 143 | MLP | medium | relu | adam | mse | - | early_stopping | 92.37% | 0.52 |
| 144 | CNN | small | relu | sgd | cross_entropy | avg | dropout | 92.37% | 4.13 |
| 145 | CNN | small | relu | sgd | cross_entropy | avg | early_stopping | 92.35% | 3.97 |
| 146 | CNN | medium | leaky_relu | sgd | cross_entropy | avg | 无 | 92.35% | 9.27 |
| 147 | CNN | large | relu | sgd | mse | max | 无 | 92.35% | 16.06 |
| 148 | CNN | large | tanh | sgd | mse | max | dropout | 92.33% | 13.30 |
| 149 | MLP | small | relu | adam | cross_entropy | - | dropout | 92.32% | 0.40 |
| 150 | CNN | medium | tanh | sgd | cross_entropy | avg | early_stopping | 92.32% | 8.70 |
| 151 | CNN | large | leaky_relu | sgd | mse | max | early_stopping | 92.32% | 13.74 |
| 152 | MLP | small | sigmoid | adam | mse | - | 无 | 92.28% | 0.34 |
| 153 | MLP | small | tanh | adam | mse | - | early_stopping | 92.27% | 0.34 |
| 154 | MLP | large | relu | adam | mse | - | 无 | 92.25% | 1.05 |
| 155 | CNN | medium | relu | sgd | cross_entropy | avg | 无 | 92.23% | 8.05 |
| 156 | CNN | large | tanh | sgd | mse | max | 无 | 92.23% | 13.08 |
| 157 | CNN | small | tanh | adam | cross_entropy | avg | early_stopping | 92.22% | 4.66 |
| 158 | CNN | large | leaky_relu | sgd | cross_entropy | avg | dropout, early_stopping | 92.22% | 11.51 |
| 159 | CNN | large | tanh | sgd | mse | max | dropout, early_stopping | 92.19% | 13.35 |
| 160 | MLP | medium | relu | adam | cross_entropy | - | 无 | 92.18% | 0.60 |
| 161 | CNN | small | tanh | sgd | cross_entropy | avg | early_stopping | 92.13% | 5.13 |
| 162 | CNN | small | leaky_relu | sgd | cross_entropy | avg | early_stopping | 92.08% | 4.33 |
| 163 | CNN | large | tanh | sgd | mse | max | early_stopping | 92.05% | 13.05 |
| 164 | MLP | medium | sigmoid | adam | mse | - | dropout, early_stopping | 92.01% | 0.61 |
| 165 | MLP | large | sigmoid | adam | cross_entropy | - | early_stopping | 92.01% | 1.09 |
| 166 | CNN | medium | leaky_relu | sgd | cross_entropy | avg | early_stopping | 91.99% | 6.46 |
| 167 | CNN | small | tanh | sgd | cross_entropy | avg | 无 | 91.98% | 4.92 |
| 168 | CNN | large | relu | adam | mse | max | early_stopping | 91.93% | 14.02 |
| 169 | CNN | large | relu | adam | mse | avg | dropout, early_stopping | 91.92% | 13.89 |
| 170 | MLP | small | tanh | adam | cross_entropy | - | dropout | 91.84% | 0.36 |
| 171 | MLP | medium | sigmoid | adam | cross_entropy | - | 无 | 91.84% | 0.57 |
| 172 | MLP | small | sigmoid | adam | mse | - | dropout | 91.82% | 0.39 |
| 173 | MLP | large | sigmoid | adam | cross_entropy | - | dropout, early_stopping | 91.82% | 1.21 |
| 174 | MLP | large | sigmoid | adam | cross_entropy | - | 无 | 91.81% | 1.06 |
| 175 | MLP | medium | relu | adam | mse | - | 无 | 91.77% | 0.54 |
| 176 | MLP | medium | relu | adam | cross_entropy | - | dropout | 91.74% | 0.60 |
| 177 | MLP | medium | sigmoid | adam | mse | - | dropout | 91.73% | 0.56 |
| 178 | CNN | large | relu | adam | mse | avg | 无 | 91.72% | 16.59 |
| 179 | MLP | small | leaky_relu | adam | cross_entropy | - | dropout, early_stopping | 91.70% | 0.37 |
| 180 | MLP | small | tanh | adam | cross_entropy | - | dropout, early_stopping | 91.69% | 0.38 |
| 181 | MLP | large | sigmoid | adam | cross_entropy | - | dropout | 91.68% | 1.23 |
| 182 | CNN | large | leaky_relu | adam | mse | avg | 无 | 91.68% | 14.58 |
| 183 | CNN | small | relu | adam | cross_entropy | max | early_stopping | 91.66% | 4.21 |
| 184 | MLP | large | leaky_relu | adam | mse | - | early_stopping | 91.65% | 1.07 |
| 185 | CNN | large | leaky_relu | adam | cross_entropy | max | dropout, early_stopping | 91.65% | 14.24 |
| 186 | MLP | medium | tanh | adam | mse | - | 无 | 91.64% | 0.57 |
| 187 | MLP | small | leaky_relu | adam | mse | - | early_stopping | 91.61% | 0.32 |
| 188 | CNN | small | tanh | sgd | cross_entropy | avg | dropout, early_stopping | 91.60% | 4.67 |
| 189 | CNN | medium | leaky_relu | adam | mse | avg | dropout | 91.60% | 6.70 |
| 190 | MLP | medium | sigmoid | adam | cross_entropy | - | early_stopping | 91.59% | 0.65 |
| 191 | MLP | small | tanh | adam | cross_entropy | - | 无 | 91.56% | 0.36 |
| 192 | CNN | small | tanh | sgd | cross_entropy | avg | dropout | 91.56% | 4.62 |
| 193 | MLP | medium | leaky_relu | adam | mse | - | early_stopping | 91.55% | 0.54 |
| 194 | MLP | medium | tanh | adam | mse | - | early_stopping | 91.53% | 0.59 |
| 195 | MLP | medium | tanh | adam | mse | - | dropout, early_stopping | 91.49% | 0.60 |
| 196 | CNN | small | tanh | adam | cross_entropy | avg | dropout, early_stopping | 91.48% | 5.02 |
| 197 | MLP | medium | relu | adam | cross_entropy | - | early_stopping | 91.47% | 0.62 |
| 198 | MLP | medium | leaky_relu | sgd | cross_entropy | - | 无 | 91.46% | 0.41 |
| 199 | CNN | small | tanh | adam | cross_entropy | max | dropout | 91.44% | 5.20 |
| 200 | MLP | small | relu | adam | cross_entropy | - | dropout, early_stopping | 91.41% | 0.37 |
| 201 | MLP | medium | relu | adam | cross_entropy | - | dropout, early_stopping | 91.36% | 0.61 |
| 202 | CNN | small | leaky_relu | sgd | cross_entropy | avg | 无 | 91.32% | 4.36 |
| 203 | CNN | medium | tanh | sgd | cross_entropy | avg | dropout | 91.29% | 9.47 |
| 204 | MLP | small | relu | sgd | cross_entropy | - | dropout, early_stopping | 91.28% | 0.29 |
| 205 | MLP | medium | relu | sgd | cross_entropy | - | early_stopping | 91.28% | 0.41 |
| 206 | MLP | medium | tanh | adam | mse | - | dropout | 91.27% | 0.59 |
| 207 | CNN | large | relu | sgd | cross_entropy | avg | dropout, early_stopping | 91.25% | 16.76 |
| 208 | CNN | medium | tanh | sgd | cross_entropy | avg | dropout, early_stopping | 91.24% | 8.97 |
| 209 | CNN | large | relu | sgd | cross_entropy | avg | 无 | 91.16% | 12.11 |
| 210 | MLP | small | relu | sgd | cross_entropy | - | dropout | 91.10% | 0.29 |
| 211 | CNN | large | leaky_relu | sgd | mse | max | dropout | 91.09% | 13.76 |
| 212 | CNN | small | relu | sgd | cross_entropy | avg | dropout, early_stopping | 91.05% | 3.92 |
| 213 | CNN | large | relu | adam | mse | max | dropout | 91.05% | 17.07 |
| 214 | MLP | small | leaky_relu | sgd | cross_entropy | - | dropout, early_stopping | 90.97% | 0.25 |
| 215 | MLP | medium | leaky_relu | sgd | cross_entropy | - | early_stopping | 90.97% | 0.36 |
| 216 | MLP | medium | leaky_relu | adam | cross_entropy | - | dropout, early_stopping | 90.96% | 0.67 |
| 217 | MLP | medium | leaky_relu | sgd | cross_entropy | - | dropout | 90.91% | 0.42 |
| 218 | MLP | medium | leaky_relu | adam | cross_entropy | - | dropout | 90.90% | 0.61 |
| 219 | MLP | medium | relu | sgd | cross_entropy | - | dropout | 90.86% | 0.44 |
| 220 | CNN | large | leaky_relu | sgd | mse | max | 无 | 90.86% | 13.91 |
| 221 | MLP | small | relu | sgd | cross_entropy | - | 无 | 90.82% | 0.26 |
| 222 | CNN | small | leaky_relu | sgd | cross_entropy | avg | dropout, early_stopping | 90.82% | 4.44 |
| 223 | MLP | small | tanh | sgd | cross_entropy | - | early_stopping | 90.79% | 0.27 |
| 224 | CNN | medium | tanh | sgd | mse | max | dropout, early_stopping | 90.76% | 9.24 |
| 225 | MLP | medium | relu | sgd | cross_entropy | - | 无 | 90.73% | 0.40 |
| 226 | MLP | small | leaky_relu | sgd | cross_entropy | - | early_stopping | 90.71% | 0.24 |
| 227 | MLP | medium | leaky_relu | adam | cross_entropy | - | 无 | 90.71% | 0.53 |
| 228 | MLP | small | leaky_relu | sgd | cross_entropy | - | dropout | 90.70% | 0.26 |
| 229 | CNN | medium | relu | adam | mse | max | early_stopping | 90.70% | 8.57 |
| 230 | MLP | medium | tanh | sgd | cross_entropy | - | early_stopping | 90.69% | 0.41 |
| 231 | MLP | small | relu | sgd | cross_entropy | - | early_stopping | 90.63% | 0.24 |
| 232 | MLP | medium | tanh | sgd | cross_entropy | - | dropout | 90.63% | 0.46 |
| 233 | MLP | medium | leaky_relu | sgd | cross_entropy | - | dropout, early_stopping | 90.62% | 0.42 |
| 234 | CNN | large | leaky_relu | adam | mse | avg | dropout, early_stopping | 90.58% | 15.42 |
| 235 | MLP | small | tanh | sgd | cross_entropy | - | dropout | 90.56% | 0.29 |
| 236 | MLP | large | leaky_relu | sgd | cross_entropy | - | early_stopping | 90.48% | 0.79 |
| 237 | MLP | small | tanh | sgd | cross_entropy | - | dropout, early_stopping | 90.47% | 0.29 |
| 238 | CNN | medium | relu | adam | cross_entropy | max | dropout, early_stopping | 90.47% | 9.09 |
| 239 | CNN | small | tanh | adam | cross_entropy | max | dropout, early_stopping | 90.46% | 4.92 |
| 240 | MLP | large | relu | sgd | cross_entropy | - | dropout | 90.43% | 0.91 |
| 241 | CNN | medium | tanh | adam | cross_entropy | max | 无 | 90.43% | 9.51 |
| 242 | MLP | small | relu | adam | mse | - | early_stopping | 90.42% | 0.36 |
| 243 | MLP | small | leaky_relu | sgd | cross_entropy | - | 无 | 90.38% | 0.24 |
| 244 | CNN | medium | leaky_relu | sgd | mse | max | early_stopping | 90.38% | 6.32 |
| 245 | CNN | medium | tanh | sgd | mse | max | 无 | 90.36% | 9.18 |
| 246 | MLP | large | relu | adam | mse | - | early_stopping | 90.31% | 1.10 |
| 247 | CNN | large | relu | adam | mse | avg | dropout | 90.31% | 16.62 |
| 248 | MLP | medium | relu | sgd | cross_entropy | - | dropout, early_stopping | 90.30% | 0.42 |
| 249 | MLP | medium | tanh | sgd | cross_entropy | - | dropout, early_stopping | 90.28% | 0.46 |
| 250 | MLP | large | relu | sgd | cross_entropy | - | 无 | 90.28% | 0.78 |
| 251 | MLP | small | tanh | sgd | cross_entropy | - | 无 | 90.26% | 0.25 |
| 252 | CNN | large | relu | sgd | mse | max | early_stopping | 90.18% | 17.09 |
| 253 | MLP | large | tanh | sgd | cross_entropy | - | 无 | 90.12% | 0.76 |
| 254 | CNN | large | relu | adam | mse | max | dropout, early_stopping | 90.11% | 13.44 |
| 255 | MLP | small | relu | adam | mse | - | 无 | 90.10% | 0.35 |
| 256 | MLP | large | tanh | sgd | cross_entropy | - | early_stopping | 90.10% | 0.74 |
| 257 | CNN | medium | relu | sgd | mse | max | dropout | 90.07% | 8.46 |
| 258 | CNN | medium | tanh | sgd | mse | max | dropout | 90.03% | 9.43 |
| 259 | MLP | small | leaky_relu | adam | mse | - | dropout, early_stopping | 89.99% | 0.36 |
| 260 | MLP | large | leaky_relu | sgd | cross_entropy | - | dropout | 89.99% | 0.87 |
| 261 | CNN | large | relu | sgd | mse | max | dropout, early_stopping | 89.95% | 12.60 |
| 262 | CNN | large | relu | sgd | mse | max | dropout | 89.92% | 16.07 |
| 263 | CNN | large | leaky_relu | sgd | mse | max | dropout, early_stopping | 89.87% | 14.01 |
| 264 | MLP | large | relu | sgd | cross_entropy | - | dropout, early_stopping | 89.82% | 0.89 |
| 265 | CNN | small | tanh | adam | cross_entropy | avg | dropout | 89.79% | 4.86 |
| 266 | CNN | medium | leaky_relu | sgd | mse | max | dropout | 89.78% | 6.79 |
| 267 | MLP | medium | tanh | sgd | cross_entropy | - | 无 | 89.77% | 0.38 |
| 268 | MLP | large | tanh | sgd | cross_entropy | - | dropout, early_stopping | 89.77% | 0.86 |
| 269 | CNN | medium | relu | sgd | mse | max | early_stopping | 89.71% | 8.80 |
| 270 | CNN | medium | relu | sgd | mse | max | 无 | 89.68% | 8.59 |
| 271 | MLP | large | sigmoid | adam | mse | - | 无 | 89.61% | 1.15 |
| 272 | MLP | medium | relu | adam | mse | - | dropout | 89.47% | 0.58 |
| 273 | MLP | large | leaky_relu | sgd | cross_entropy | - | dropout, early_stopping | 89.44% | 0.88 |
| 274 | MLP | small | tanh | adam | mse | - | dropout, early_stopping | 89.43% | 0.35 |
| 275 | CNN | medium | tanh | sgd | mse | max | early_stopping | 89.42% | 9.13 |
| 276 | MLP | large | relu | adam | cross_entropy | - | dropout, early_stopping | 89.33% | 1.18 |
| 277 | MLP | large | leaky_relu | adam | cross_entropy | - | early_stopping | 89.29% | 1.09 |
| 278 | CNN | medium | relu | sgd | mse | max | dropout, early_stopping | 89.24% | 8.37 |
| 279 | MLP | medium | tanh | adam | cross_entropy | - | dropout | 89.21% | 0.58 |
| 280 | CNN | medium | tanh | adam | cross_entropy | max | dropout | 89.21% | 9.23 |
| 281 | MLP | medium | leaky_relu | adam | mse | - | dropout, early_stopping | 89.15% | 0.57 |
| 282 | MLP | large | leaky_relu | sgd | cross_entropy | - | 无 | 89.00% | 0.77 |
| 283 | MLP | large | leaky_relu | adam | mse | - | dropout, early_stopping | 88.87% | 1.24 |
| 284 | CNN | small | tanh | adam | cross_entropy | avg | 无 | 88.83% | 4.92 |
| 285 | CNN | medium | leaky_relu | sgd | mse | max | 无 | 88.81% | 6.46 |
| 286 | MLP | medium | tanh | adam | cross_entropy | - | early_stopping | 88.75% | 0.54 |
| 287 | MLP | large | relu | adam | cross_entropy | - | dropout | 88.72% | 1.17 |
| 288 | MLP | small | tanh | adam | mse | - | dropout | 88.66% | 0.41 |
| 289 | CNN | small | tanh | adam | mse | avg | early_stopping | 88.49% | 5.08 |
| 290 | MLP | large | relu | sgd | cross_entropy | - | early_stopping | 88.38% | 0.81 |
| 291 | MLP | medium | tanh | adam | cross_entropy | - | 无 | 88.36% | 0.59 |
| 292 | MLP | large | tanh | sgd | cross_entropy | - | dropout | 88.15% | 0.93 |
| 293 | MLP | medium | tanh | adam | cross_entropy | - | dropout, early_stopping | 88.05% | 0.63 |
| 294 | CNN | small | tanh | adam | mse | max | 无 | 88.01% | 5.04 |
| 295 | CNN | small | tanh | sgd | mse | max | dropout, early_stopping | 87.87% | 4.91 |
| 296 | CNN | medium | leaky_relu | sgd | mse | max | dropout, early_stopping | 87.78% | 6.50 |
| 297 | CNN | small | tanh | sgd | mse | max | 无 | 87.66% | 6.01 |
| 298 | MLP | small | leaky_relu | adam | mse | - | dropout | 87.61% | 0.36 |
| 299 | MLP | small | relu | adam | mse | - | dropout, early_stopping | 87.46% | 0.36 |
| 300 | MLP | medium | relu | adam | mse | - | dropout, early_stopping | 87.36% | 0.61 |
| 301 | MLP | large | relu | adam | mse | - | dropout | 87.31% | 1.23 |
| 302 | CNN | small | tanh | sgd | mse | max | dropout | 87.27% | 6.13 |
| 303 | CNN | small | relu | sgd | mse | max | early_stopping | 87.22% | 3.87 |
| 304 | CNN | medium | tanh | adam | cross_entropy | max | dropout, early_stopping | 87.07% | 9.70 |
| 305 | CNN | small | tanh | sgd | mse | max | early_stopping | 86.96% | 4.99 |
| 306 | CNN | small | leaky_relu | sgd | mse | max | 无 | 86.83% | 4.41 |
| 307 | MLP | medium | leaky_relu | adam | cross_entropy | - | early_stopping | 86.80% | 0.55 |
| 308 | CNN | small | relu | sgd | mse | max | dropout, early_stopping | 86.80% | 3.98 |
| 309 | MLP | small | relu | sgd | mse | - | early_stopping | 86.78% | 0.24 |
| 310 | CNN | small | leaky_relu | sgd | mse | max | dropout | 86.78% | 4.55 |
| 311 | MLP | medium | leaky_relu | adam | mse | - | dropout | 86.73% | 0.60 |
| 312 | MLP | small | sigmoid | sgd | cross_entropy | - | 无 | 86.68% | 0.24 |
| 313 | CNN | small | leaky_relu | sgd | mse | max | early_stopping | 86.61% | 4.50 |
| 314 | MLP | small | relu | adam | mse | - | dropout | 86.60% | 0.35 |
| 315 | MLP | large | tanh | adam | cross_entropy | - | dropout | 86.58% | 1.33 |
| 316 | MLP | large | tanh | adam | cross_entropy | - | dropout, early_stopping | 86.57% | 1.26 |
| 317 | MLP | small | relu | sgd | mse | - | 无 | 86.55% | 0.24 |
| 318 | MLP | large | relu | adam | mse | - | dropout, early_stopping | 86.55% | 1.20 |
| 319 | CNN | medium | relu | adam | cross_entropy | avg | dropout, early_stopping | 86.29% | 8.62 |
| 320 | MLP | small | leaky_relu | sgd | mse | - | 无 | 86.16% | 0.32 |
| 321 | CNN | large | leaky_relu | sgd | mse | avg | dropout | 86.13% | 13.62 |
| 322 | MLP | large | tanh | adam | cross_entropy | - | 无 | 86.11% | 1.13 |
| 323 | MLP | small | relu | sgd | mse | - | dropout | 86.10% | 0.26 |
| 324 | MLP | small | leaky_relu | sgd | mse | - | dropout | 86.08% | 0.25 |
| 325 | MLP | small | leaky_relu | sgd | mse | - | early_stopping | 86.06% | 0.21 |
| 326 | CNN | small | relu | sgd | mse | max | dropout | 86.01% | 3.96 |
| 327 | MLP | large | tanh | adam | cross_entropy | - | early_stopping | 85.87% | 1.15 |
| 328 | CNN | small | relu | sgd | mse | max | 无 | 85.85% | 4.10 |
| 329 | CNN | large | tanh | sgd | mse | avg | 无 | 85.77% | 13.06 |
| 330 | MLP | small | sigmoid | sgd | cross_entropy | - | dropout | 85.70% | 0.29 |
| 331 | CNN | small | leaky_relu | sgd | mse | max | dropout, early_stopping | 85.65% | 4.75 |
| 332 | CNN | large | leaky_relu | sgd | mse | avg | dropout, early_stopping | 85.64% | 13.85 |
| 333 | MLP | small | sigmoid | sgd | cross_entropy | - | early_stopping | 85.61% | 0.25 |
| 334 | CNN | large | leaky_relu | sgd | mse | avg | 无 | 85.36% | 13.62 |
| 335 | MLP | small | leaky_relu | sgd | mse | - | dropout, early_stopping | 85.28% | 0.26 |
| 336 | CNN | large | leaky_relu | sgd | mse | avg | early_stopping | 85.28% | 13.56 |
| 337 | CNN | large | tanh | sgd | mse | avg | dropout, early_stopping | 85.25% | 13.13 |
| 338 | CNN | medium | relu | sgd | mse | avg | dropout | 85.03% | 8.54 |
| 339 | CNN | medium | tanh | sgd | mse | avg | 无 | 84.86% | 8.80 |
| 340 | MLP | small | relu | sgd | mse | - | dropout, early_stopping | 84.84% | 0.24 |
| 341 | CNN | large | tanh | sgd | mse | avg | early_stopping | 84.75% | 13.37 |
| 342 | MLP | small | sigmoid | sgd | cross_entropy | - | dropout, early_stopping | 84.70% | 0.28 |
| 343 | CNN | large | relu | sgd | mse | avg | dropout | 84.67% | 16.07 |
| 344 | MLP | small | tanh | sgd | mse | - | dropout, early_stopping | 84.66% | 0.26 |
| 345 | MLP | small | tanh | sgd | mse | - | early_stopping | 84.57% | 0.24 |
| 346 | CNN | medium | leaky_relu | sgd | mse | avg | 无 | 84.56% | 6.31 |
| 347 | MLP | small | tanh | sgd | mse | - | 无 | 84.12% | 0.25 |
| 348 | MLP | medium | tanh | sgd | mse | - | early_stopping | 84.08% | 0.37 |
| 349 | CNN | large | tanh | sgd | mse | avg | dropout | 84.01% | 13.24 |
| 350 | CNN | medium | leaky_relu | sgd | mse | avg | early_stopping | 83.97% | 6.84 |
| 351 | MLP | small | tanh | sgd | mse | - | dropout | 83.91% | 0.26 |
| 352 | CNN | medium | tanh | sgd | mse | avg | dropout | 83.89% | 8.81 |
| 353 | MLP | medium | tanh | sgd | mse | - | 无 | 83.65% | 0.40 |
| 354 | CNN | medium | leaky_relu | sgd | mse | avg | dropout | 83.62% | 6.48 |
| 355 | MLP | medium | relu | sgd | mse | - | 无 | 83.58% | 0.37 |
| 356 | CNN | medium | tanh | sgd | mse | avg | dropout, early_stopping | 83.46% | 8.79 |
| 357 | CNN | large | relu | sgd | mse | avg | dropout, early_stopping | 83.39% | 12.39 |
| 358 | CNN | medium | tanh | sgd | mse | avg | early_stopping | 83.30% | 8.91 |
| 359 | MLP | medium | tanh | sgd | mse | - | dropout | 83.24% | 0.45 |
| 360 | CNN | large | relu | sgd | mse | avg | 无 | 83.05% | 13.72 |
| 361 | CNN | small | tanh | sgd | mse | avg | 无 | 82.92% | 4.88 |
| 362 | MLP | medium | leaky_relu | sgd | mse | - | early_stopping | 82.91% | 0.34 |
| 363 | CNN | small | tanh | sgd | mse | avg | dropout | 82.89% | 5.40 |
| 364 | MLP | medium | tanh | sgd | mse | - | dropout, early_stopping | 82.61% | 0.42 |
| 365 | CNN | small | tanh | sgd | mse | avg | early_stopping | 82.59% | 4.82 |
| 366 | CNN | medium | relu | sgd | mse | avg | early_stopping | 82.58% | 8.13 |
| 367 | CNN | small | leaky_relu | sgd | mse | avg | 无 | 82.56% | 4.14 |
| 368 | MLP | large | tanh | sgd | mse | - | early_stopping | 82.42% | 1.13 |
| 369 | CNN | small | tanh | sgd | mse | avg | dropout, early_stopping | 82.28% | 4.92 |
| 370 | CNN | large | relu | sgd | mse | avg | early_stopping | 82.18% | 16.25 |
| 371 | MLP | large | tanh | sgd | mse | - | 无 | 82.11% | 0.84 |
| 372 | CNN | medium | relu | sgd | mse | avg | 无 | 82.10% | 8.43 |
| 373 | MLP | medium | relu | sgd | mse | - | early_stopping | 82.04% | 0.37 |
| 374 | CNN | small | tanh | adam | mse | avg | dropout | 81.83% | 6.18 |
| 375 | MLP | medium | leaky_relu | sgd | mse | - | 无 | 81.82% | 0.36 |
| 376 | CNN | medium | leaky_relu | sgd | mse | avg | dropout, early_stopping | 81.71% | 6.82 |
| 377 | CNN | medium | relu | sgd | mse | avg | dropout, early_stopping | 81.68% | 8.83 |
| 378 | MLP | large | tanh | sgd | mse | - | dropout | 81.16% | 2.01 |
| 379 | CNN | small | relu | sgd | mse | avg | dropout | 81.14% | 3.94 |
| 380 | MLP | large | leaky_relu | adam | cross_entropy | - | 无 | 80.81% | 1.08 |
| 381 | CNN | small | leaky_relu | sgd | mse | avg | early_stopping | 80.80% | 4.14 |
| 382 | CNN | small | relu | sgd | mse | avg | early_stopping | 80.52% | 3.98 |
| 383 | MLP | medium | leaky_relu | sgd | mse | - | dropout | 80.26% | 0.44 |
| 384 | MLP | medium | relu | sgd | mse | - | dropout, early_stopping | 80.24% | 0.45 |
| 385 | CNN | small | relu | sgd | mse | avg | dropout, early_stopping | 80.22% | 3.95 |
| 386 | MLP | large | tanh | sgd | mse | - | dropout, early_stopping | 79.75% | 1.05 |
| 387 | CNN | medium | tanh | adam | cross_entropy | avg | early_stopping | 79.50% | 9.05 |
| 388 | CNN | small | relu | sgd | mse | avg | 无 | 79.27% | 3.95 |
| 389 | MLP | large | leaky_relu | adam | cross_entropy | - | dropout | 79.21% | 1.22 |
| 390 | MLP | medium | leaky_relu | sgd | mse | - | dropout, early_stopping | 78.63% | 0.40 |
| 391 | MLP | medium | relu | sgd | mse | - | dropout | 78.21% | 0.43 |
| 392 | CNN | small | leaky_relu | sgd | mse | avg | dropout, early_stopping | 77.21% | 4.39 |
| 393 | MLP | large | leaky_relu | adam | cross_entropy | - | dropout, early_stopping | 77.11% | 1.19 |
| 394 | CNN | small | leaky_relu | sgd | mse | avg | dropout | 76.89% | 4.32 |
| 395 | MLP | large | relu | sgd | mse | - | 无 | 73.19% | 0.74 |
| 396 | MLP | large | leaky_relu | sgd | mse | - | 无 | 72.95% | 0.73 |
| 397 | MLP | large | leaky_relu | adam | mse | - | dropout | 72.65% | 1.18 |
| 398 | MLP | small | sigmoid | sgd | mse | - | early_stopping | 71.95% | 0.23 |
| 399 | MLP | large | leaky_relu | sgd | mse | - | early_stopping | 71.91% | 0.80 |
| 400 | MLP | large | sigmoid | adam | mse | - | dropout | 71.67% | 1.25 |
| 401 | MLP | small | sigmoid | sgd | mse | - | 无 | 71.28% | 0.23 |
| 402 | MLP | small | sigmoid | sgd | mse | - | dropout, early_stopping | 71.07% | 0.31 |
| 403 | MLP | large | leaky_relu | sgd | mse | - | dropout | 71.03% | 0.81 |
| 404 | MLP | large | relu | sgd | mse | - | early_stopping | 69.02% | 0.75 |
| 405 | MLP | large | relu | sgd | mse | - | dropout, early_stopping | 67.29% | 0.84 |
| 406 | MLP | large | relu | sgd | mse | - | dropout | 66.96% | 0.86 |
| 407 | MLP | large | leaky_relu | sgd | mse | - | dropout, early_stopping | 64.23% | 0.83 |
| 408 | MLP | large | sigmoid | adam | mse | - | dropout, early_stopping | 56.47% | 1.22 |
| 409 | MLP | small | sigmoid | sgd | mse | - | dropout | 55.27% | 0.26 |
| 410 | CNN | large | tanh | adam | cross_entropy | max | dropout, early_stopping | 52.77% | 13.90 |
| 411 | MLP | large | tanh | adam | mse | - | early_stopping | 47.68% | 1.18 |
| 412 | MLP | large | tanh | adam | mse | - | 无 | 45.97% | 1.09 |
| 413 | MLP | medium | sigmoid | sgd | cross_entropy | - | early_stopping | 37.39% | 0.39 |
| 414 | MLP | medium | sigmoid | sgd | cross_entropy | - | 无 | 35.39% | 0.39 |
| 415 | MLP | medium | sigmoid | sgd | mse | - | early_stopping | 32.07% | 0.38 |
| 416 | MLP | large | tanh | adam | mse | - | dropout, early_stopping | 28.03% | 1.25 |
| 417 | MLP | medium | sigmoid | sgd | cross_entropy | - | dropout, early_stopping | 24.79% | 0.43 |
| 418 | MLP | medium | sigmoid | sgd | cross_entropy | - | dropout | 24.25% | 0.44 |
| 419 | MLP | large | tanh | adam | mse | - | dropout | 21.07% | 1.29 |
| 420 | MLP | medium | sigmoid | sgd | mse | - | dropout | 20.48% | 0.43 |
| 421 | MLP | medium | sigmoid | sgd | mse | - | 无 | 19.18% | 0.38 |
| 422 | CNN | large | sigmoid | sgd | mse | avg | early_stopping | 16.13% | 12.08 |
| 423 | CNN | small | sigmoid | sgd | mse | avg | dropout | 11.36% | 4.18 |
| 424 | MLP | large | sigmoid | sgd | cross_entropy | - | 无 | 11.35% | 0.77 |
| 425 | MLP | large | sigmoid | sgd | cross_entropy | - | dropout | 11.35% | 0.87 |
| 426 | MLP | large | sigmoid | sgd | cross_entropy | - | early_stopping | 11.35% | 0.76 |
| 427 | MLP | large | sigmoid | sgd | mse | - | dropout | 11.35% | 0.86 |
| 428 | CNN | small | relu | adam | mse | max | 无 | 11.35% | 4.17 |
| 429 | CNN | small | relu | adam | mse | max | dropout | 11.35% | 4.22 |
| 430 | CNN | small | sigmoid | sgd | cross_entropy | max | 无 | 11.35% | 4.06 |
| 431 | CNN | small | sigmoid | sgd | cross_entropy | max | dropout | 11.35% | 4.31 |
| 432 | CNN | small | sigmoid | sgd | cross_entropy | avg | dropout | 11.35% | 4.16 |
| 433 | CNN | small | sigmoid | sgd | cross_entropy | max | dropout, early_stopping | 11.35% | 4.32 |
| 434 | CNN | small | sigmoid | sgd | cross_entropy | avg | dropout, early_stopping | 11.35% | 4.25 |
| 435 | CNN | small | sigmoid | adam | cross_entropy | max | dropout | 11.35% | 4.34 |
| 436 | CNN | small | sigmoid | adam | cross_entropy | max | dropout, early_stopping | 11.35% | 4.25 |
| 437 | CNN | small | sigmoid | sgd | mse | avg | 无 | 11.35% | 4.08 |
| 438 | CNN | small | sigmoid | sgd | mse | max | dropout, early_stopping | 11.35% | 4.02 |
| 439 | CNN | small | sigmoid | adam | mse | avg | dropout | 11.35% | 4.51 |
| 440 | CNN | small | tanh | adam | mse | avg | dropout, early_stopping | 11.35% | 4.99 |
| 441 | CNN | medium | relu | adam | cross_entropy | max | 无 | 11.35% | 8.71 |
| 442 | CNN | medium | relu | adam | cross_entropy | avg | 无 | 11.35% | 8.85 |
| 443 | CNN | medium | sigmoid | sgd | cross_entropy | max | 无 | 11.35% | 8.73 |
| 444 | CNN | medium | sigmoid | sgd | cross_entropy | avg | 无 | 11.35% | 8.22 |
| 445 | CNN | medium | sigmoid | sgd | cross_entropy | avg | dropout | 11.35% | 8.86 |
| 446 | CNN | medium | sigmoid | sgd | cross_entropy | avg | early_stopping | 11.35% | 8.37 |
| 447 | CNN | medium | sigmoid | sgd | cross_entropy | max | dropout, early_stopping | 11.35% | 8.30 |
| 448 | CNN | medium | sigmoid | sgd | cross_entropy | avg | dropout, early_stopping | 11.35% | 8.59 |
| 449 | CNN | medium | sigmoid | adam | cross_entropy | max | dropout | 11.35% | 8.54 |
| 450 | CNN | medium | sigmoid | adam | cross_entropy | max | dropout, early_stopping | 11.35% | 8.96 |
| 451 | CNN | medium | sigmoid | sgd | mse | avg | 无 | 11.35% | 8.54 |
| 452 | CNN | medium | sigmoid | sgd | mse | avg | dropout | 11.35% | 8.26 |
| 453 | CNN | medium | sigmoid | sgd | mse | avg | early_stopping | 11.35% | 8.59 |
| 454 | CNN | medium | tanh | adam | cross_entropy | avg | 无 | 11.35% | 9.15 |
| 455 | CNN | medium | tanh | adam | cross_entropy | avg | dropout | 11.35% | 9.82 |
| 456 | CNN | large | relu | adam | cross_entropy | max | 无 | 11.35% | 16.77 |
| 457 | CNN | large | relu | adam | cross_entropy | max | dropout, early_stopping | 11.35% | 14.95 |
| 458 | CNN | large | sigmoid | sgd | cross_entropy | max | 无 | 11.35% | 12.54 |
| 459 | CNN | large | sigmoid | sgd | cross_entropy | avg | 无 | 11.35% | 12.40 |
| 460 | CNN | large | sigmoid | sgd | cross_entropy | max | dropout | 11.35% | 12.71 |
| 461 | CNN | large | sigmoid | sgd | cross_entropy | avg | dropout | 11.35% | 13.36 |
| 462 | CNN | large | sigmoid | sgd | cross_entropy | avg | early_stopping | 11.35% | 12.54 |
| 463 | CNN | large | sigmoid | sgd | cross_entropy | max | dropout, early_stopping | 11.35% | 12.56 |
| 464 | CNN | large | sigmoid | sgd | cross_entropy | avg | dropout, early_stopping | 11.35% | 12.39 |
| 465 | CNN | large | sigmoid | adam | cross_entropy | avg | early_stopping | 11.35% | 13.07 |
| 466 | CNN | large | sigmoid | sgd | mse | max | dropout, early_stopping | 11.35% | 12.11 |
| 467 | CNN | large | sigmoid | adam | mse | max | dropout | 11.35% | 12.83 |
| 468 | CNN | large | tanh | adam | mse | max | dropout, early_stopping | 11.35% | 15.00 |
| 469 | MLP | large | sigmoid | sgd | mse | - | 无 | 10.66% | 0.77 |
| 470 | CNN | medium | sigmoid | sgd | mse | max | 无 | 10.55% | 8.20 |
| 471 | MLP | large | sigmoid | sgd | cross_entropy | - | dropout, early_stopping | 10.32% | 0.91 |
| 472 | CNN | small | relu | adam | mse | avg | dropout | 10.32% | 4.26 |
| 473 | CNN | medium | relu | adam | mse | avg | 无 | 10.32% | 9.06 |
| 474 | CNN | medium | sigmoid | adam | cross_entropy | avg | 无 | 10.32% | 9.24 |
| 475 | CNN | medium | sigmoid | adam | cross_entropy | max | early_stopping | 10.32% | 9.17 |
| 476 | CNN | medium | sigmoid | adam | mse | max | 无 | 10.32% | 8.62 |
| 477 | CNN | medium | sigmoid | adam | mse | avg | 无 | 10.32% | 9.10 |
| 478 | CNN | large | sigmoid | adam | cross_entropy | max | 无 | 10.32% | 13.33 |
| 479 | CNN | large | sigmoid | adam | mse | avg | dropout, early_stopping | 10.32% | 12.62 |
| 480 | CNN | large | tanh | adam | cross_entropy | max | 无 | 10.32% | 18.68 |
| 481 | CNN | large | tanh | adam | cross_entropy | avg | dropout, early_stopping | 10.32% | 13.83 |
| 482 | CNN | large | tanh | adam | mse | avg | 无 | 10.32% | 15.12 |
| 483 | MLP | medium | sigmoid | sgd | mse | - | dropout, early_stopping | 10.28% | 0.43 |
| 484 | CNN | small | relu | adam | mse | avg | early_stopping | 10.28% | 4.25 |
| 485 | CNN | small | relu | adam | mse | avg | dropout, early_stopping | 10.28% | 4.27 |
| 486 | CNN | small | sigmoid | sgd | cross_entropy | avg | 无 | 10.28% | 4.12 |
| 487 | CNN | small | sigmoid | sgd | cross_entropy | max | early_stopping | 10.28% | 4.09 |
| 488 | CNN | small | sigmoid | sgd | cross_entropy | avg | early_stopping | 10.28% | 4.04 |
| 489 | CNN | small | sigmoid | adam | cross_entropy | avg | dropout | 10.28% | 4.24 |
| 490 | CNN | small | sigmoid | sgd | mse | max | early_stopping | 10.28% | 4.07 |
| 491 | CNN | small | sigmoid | adam | mse | max | early_stopping | 10.28% | 4.90 |
| 492 | CNN | medium | sigmoid | sgd | cross_entropy | max | early_stopping | 10.28% | 9.01 |
| 493 | CNN | medium | sigmoid | sgd | mse | avg | dropout, early_stopping | 10.28% | 8.28 |
| 494 | CNN | medium | tanh | adam | mse | max | 无 | 10.28% | 9.37 |
| 495 | CNN | medium | tanh | adam | mse | max | dropout, early_stopping | 10.28% | 10.10 |
| 496 | CNN | large | sigmoid | sgd | cross_entropy | max | early_stopping | 10.28% | 12.68 |
| 497 | CNN | large | sigmoid | adam | cross_entropy | max | dropout | 10.28% | 13.26 |
| 498 | CNN | large | sigmoid | sgd | mse | avg | 无 | 10.28% | 12.49 |
| 499 | CNN | large | sigmoid | sgd | mse | max | dropout | 10.28% | 12.94 |
| 500 | CNN | large | sigmoid | adam | mse | avg | 无 | 10.28% | 12.64 |
| 501 | CNN | large | tanh | adam | mse | max | early_stopping | 10.28% | 13.98 |
| 502 | CNN | small | relu | adam | mse | avg | 无 | 10.10% | 4.12 |
| 503 | CNN | small | sigmoid | adam | mse | max | 无 | 10.10% | 5.15 |
| 504 | CNN | small | tanh | adam | mse | avg | 无 | 10.10% | 4.89 |
| 505 | CNN | medium | sigmoid | adam | cross_entropy | max | 无 | 10.10% | 9.03 |
| 506 | CNN | medium | sigmoid | adam | cross_entropy | avg | early_stopping | 10.10% | 8.82 |
| 507 | CNN | medium | tanh | adam | cross_entropy | avg | dropout, early_stopping | 10.10% | 9.54 |
| 508 | CNN | large | relu | adam | mse | avg | early_stopping | 10.10% | 12.94 |
| 509 | CNN | large | tanh | adam | mse | avg | early_stopping | 10.10% | 16.52 |
| 510 | CNN | small | relu | adam | mse | max | dropout, early_stopping | 10.09% | 4.27 |
| 511 | CNN | small | sigmoid | adam | cross_entropy | max | early_stopping | 10.09% | 4.48 |
| 512 | CNN | small | sigmoid | adam | cross_entropy | avg | dropout, early_stopping | 10.09% | 4.20 |
| 513 | CNN | small | sigmoid | adam | mse | max | dropout | 10.09% | 5.14 |
| 514 | CNN | small | sigmoid | adam | mse | max | dropout, early_stopping | 10.09% | 4.71 |
| 515 | CNN | medium | relu | adam | mse | max | dropout, early_stopping | 10.09% | 9.01 |
| 516 | CNN | medium | sigmoid | sgd | cross_entropy | max | dropout | 10.09% | 8.72 |
| 517 | CNN | medium | sigmoid | adam | cross_entropy | avg | dropout | 10.09% | 9.04 |
| 518 | CNN | medium | sigmoid | sgd | mse | max | early_stopping | 10.09% | 8.98 |
| 519 | CNN | medium | sigmoid | adam | mse | max | dropout | 10.09% | 9.31 |
| 520 | CNN | medium | sigmoid | adam | mse | avg | early_stopping | 10.09% | 9.45 |
| 521 | CNN | large | sigmoid | sgd | mse | avg | dropout | 10.09% | 13.25 |
| 522 | CNN | large | sigmoid | sgd | mse | max | early_stopping | 10.09% | 13.09 |
| 523 | CNN | large | tanh | adam | cross_entropy | avg | 无 | 10.09% | 18.20 |
| 524 | CNN | large | tanh | adam | cross_entropy | max | dropout | 10.09% | 17.48 |
| 525 | CNN | large | tanh | adam | mse | max | 无 | 10.09% | 13.85 |
| 526 | CNN | large | tanh | adam | mse | avg | dropout | 10.09% | 13.64 |
| 527 | CNN | large | tanh | adam | mse | avg | dropout, early_stopping | 10.09% | 13.73 |
| 528 | CNN | small | sigmoid | adam | cross_entropy | max | 无 | 9.82% | 4.24 |
| 529 | CNN | small | sigmoid | adam | cross_entropy | avg | 无 | 9.82% | 4.48 |
| 530 | CNN | small | sigmoid | adam | cross_entropy | avg | early_stopping | 9.82% | 4.25 |
| 531 | CNN | small | sigmoid | sgd | mse | avg | early_stopping | 9.82% | 4.03 |
| 532 | CNN | small | sigmoid | adam | mse | avg | dropout, early_stopping | 9.82% | 4.63 |
| 533 | CNN | medium | relu | adam | mse | max | 无 | 9.82% | 9.05 |
| 534 | CNN | medium | sigmoid | adam | mse | max | dropout, early_stopping | 9.82% | 8.80 |
| 535 | CNN | medium | tanh | adam | mse | avg | early_stopping | 9.82% | 10.50 |
| 536 | CNN | large | relu | adam | mse | max | 无 | 9.82% | 14.48 |
| 537 | CNN | large | sigmoid | adam | mse | max | 无 | 9.82% | 12.93 |
| 538 | CNN | large | sigmoid | adam | mse | max | early_stopping | 9.82% | 12.73 |
| 539 | CNN | large | tanh | adam | cross_entropy | avg | dropout | 9.82% | 13.81 |
| 540 | MLP | large | sigmoid | sgd | mse | - | early_stopping | 9.80% | 0.76 |
| 541 | CNN | small | relu | adam | mse | max | early_stopping | 9.80% | 4.31 |
| 542 | CNN | small | sigmoid | sgd | mse | max | 无 | 9.80% | 4.09 |
| 543 | CNN | small | sigmoid | sgd | mse | max | dropout | 9.80% | 4.14 |
| 544 | CNN | medium | sigmoid | adam | mse | max | early_stopping | 9.80% | 9.11 |
| 545 | CNN | medium | tanh | adam | mse | max | dropout | 9.80% | 10.67 |
| 546 | CNN | large | sigmoid | adam | cross_entropy | max | early_stopping | 9.80% | 13.31 |
| 547 | CNN | large | sigmoid | sgd | mse | max | 无 | 9.80% | 12.62 |
| 548 | CNN | large | sigmoid | adam | mse | avg | early_stopping | 9.80% | 12.64 |
| 549 | CNN | small | sigmoid | sgd | mse | avg | dropout, early_stopping | 9.74% | 4.48 |
| 550 | CNN | medium | relu | adam | mse | avg | dropout | 9.74% | 8.74 |
| 551 | CNN | medium | relu | adam | mse | avg | dropout, early_stopping | 9.74% | 8.62 |
| 552 | CNN | medium | sigmoid | sgd | mse | max | dropout, early_stopping | 9.74% | 8.53 |
| 553 | CNN | medium | tanh | adam | cross_entropy | max | early_stopping | 9.74% | 9.31 |
| 554 | CNN | medium | tanh | adam | mse | avg | 无 | 9.74% | 9.59 |
| 555 | CNN | medium | tanh | adam | mse | max | early_stopping | 9.74% | 9.74 |
| 556 | CNN | large | sigmoid | adam | cross_entropy | avg | 无 | 9.74% | 12.97 |
| 557 | CNN | large | sigmoid | adam | cross_entropy | avg | dropout | 9.74% | 13.12 |
| 558 | CNN | large | sigmoid | adam | cross_entropy | avg | dropout, early_stopping | 9.74% | 13.39 |
| 559 | CNN | large | sigmoid | sgd | mse | avg | dropout, early_stopping | 9.74% | 11.97 |
| 560 | CNN | large | sigmoid | adam | mse | max | dropout, early_stopping | 9.74% | 12.87 |
| 561 | CNN | large | tanh | adam | cross_entropy | max | early_stopping | 9.74% | 13.80 |
| 562 | CNN | small | tanh | adam | mse | max | early_stopping | 9.58% | 5.03 |
| 563 | CNN | medium | sigmoid | adam | cross_entropy | avg | dropout, early_stopping | 9.58% | 9.43 |
| 564 | CNN | medium | sigmoid | sgd | mse | max | dropout | 9.58% | 8.89 |
| 565 | CNN | medium | sigmoid | adam | mse | avg | dropout | 9.58% | 9.20 |
| 566 | CNN | medium | sigmoid | adam | mse | avg | dropout, early_stopping | 9.58% | 8.94 |
| 567 | CNN | medium | tanh | adam | mse | avg | dropout, early_stopping | 9.58% | 13.50 |
| 568 | CNN | large | sigmoid | adam | cross_entropy | max | dropout, early_stopping | 9.58% | 12.94 |
| 569 | CNN | large | sigmoid | adam | mse | avg | dropout | 9.58% | 14.00 |
| 570 | CNN | large | tanh | adam | cross_entropy | avg | early_stopping | 9.58% | 15.02 |
| 571 | MLP | large | sigmoid | sgd | mse | - | dropout, early_stopping | 8.92% | 0.84 |
| 572 | CNN | small | sigmoid | adam | mse | avg | 无 | 8.92% | 4.62 |
| 573 | CNN | small | sigmoid | adam | mse | avg | early_stopping | 8.92% | 4.65 |
| 574 | CNN | small | tanh | adam | mse | max | dropout | 8.92% | 4.94 |
| 575 | CNN | medium | tanh | adam | mse | avg | dropout | 8.92% | 9.93 |
| 576 | CNN | large | tanh | adam | mse | max | dropout | 8.92% | 13.93 |
