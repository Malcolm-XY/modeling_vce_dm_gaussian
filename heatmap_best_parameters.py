# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 17:49:12 2025

@author: usouu
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 原始 lambda 和 sigma
lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0])
log_lambda_labels = np.round(np.log10(lambda_values)).astype(int).astype(str)  # ['-6', '-5', ..., '0']
sigma_values = [0.05, 0.1, 0.15, 0.2, 0.3]

# accuracy_data 略，保持原样
accuracy_data = {
    "sr=1.0": np.array([
        [80.18, 69.33, 89.87, 91.08, 86.92],
        [79.06, 84.43, 91.29, 90.94, 87.36],
        [80.10, 90.76, 91.09, 90.84, 88.69],
        [81.05, 92.34, 92.02, 91.76, 84.28],
        [89.93, 93.11, 92.28, 89.31, 86.69],
        [93.03, 92.44, 89.24, 88.14, 79.89],
        [92.25, 89.97, 86.21, 82.60, 78.77]
    ]),
    "sr=0.5": np.array([
        [76.74, 70.13, 90.53, 91.10, 86.40],
        [76.95, 84.11, 91.41, 91.34, 88.34],
        [76.58, 90.32, 91.64, 91.55, 89.26],
        [78.49, 92.71, 92.39, 91.97, 84.26],
        [85.86, 93.38, 91.67, 89.52, 85.06],
        [89.82, 89.68, 86.73, 84.46, 78.55],
        [87.68, 86.01, 83.45, 80.58, 75.71]
    ]),
    "sr=0.4": np.array([
        [74.06, 69.89, 89.90, 91.08, 87.63],
        [73.60, 84.11, 91.68, 91.04, 86.74],
        [73.36, 89.32, 92.17, 91.35, 89.71],
        [75.90, 92.45, 91.84, 91.24, 82.07],
        [83.28, 92.94, 91.36, 87.25, 84.56],
        [88.72, 88.61, 84.91, 83.02, 76.78],
        [87.31, 85.93, 81.21, 77.77, 75.23]
    ]),
    "sr=0.3": np.array([
        [71.93, 69.99, 88.86, 90.93, 85.96],
        [71.96, 81.77, 90.13, 90.27, 85.73],
        [71.36, 88.87, 91.47, 91.22, 88.64],
        [72.90, 91.99, 91.59, 90.62, 78.82],
        [81.29, 92.24, 90.89, 84.80, 81.94],
        [86.43, 88.19, 83.99, 81.75, 74.12],
        [83.60, 83.68, 80.79, 77.21, 73.42]
    ]),
    "sr=0.2": np.array([
        [71.96, 69.05, 85.93, 88.45, 84.33],
        [71.50, 78.06, 87.04, 87.66, 82.30],
        [70.19, 87.45, 89.08, 89.23, 86.16],
        [72.39, 89.58, 89.82, 89.22, 73.51],
        [77.35, 90.28, 88.44, 79.71, 78.48],
        [81.11, 83.60, 80.23, 77.85, 69.88],
        [77.85, 80.40, 78.76, 75.27, 69.95]
    ]),
    "sr=0.1": np.array([
        [62.97, 63.87, 72.47, 76.05, 73.13],
        [62.92, 71.38, 74.65, 77.16, 71.62],
        [63.42, 72.53, 76.54, 77.66, 77.69],
        [64.97, 79.27, 78.09, 79.97, 69.11],
        [68.08, 81.77, 78.76, 71.47, 72.31],
        [65.09, 71.90, 73.12, 72.82, 63.90],
        [63.24, 66.47, 68.94, 68.14, 65.28]
    ])
}

# 准备绘图
vmin = 60
vmax = 95
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
axes = axes.flatten()
sr_keys = list(accuracy_data.keys())

mappable = None

for i, sr in enumerate(sr_keys):
    ax = axes[i]
    acc_matrix = accuracy_data[sr]
    show_xlabel = (i >= 3)
    show_ylabel = (i % 3 == 0)

    heatmap = sns.heatmap(
        acc_matrix,
        ax=ax,
        xticklabels=sigma_values if show_xlabel else False,
        yticklabels=log_lambda_labels if show_ylabel else False,
        cmap='coolwarm',
        annot=True,
        fmt=".1f",
        vmin=vmin,
        vmax=vmax,
        cbar=False
    )

    if show_xlabel:
        ax.set_xlabel('sigma')
    if show_ylabel:
        ax.set_ylabel('log10(lambda)')  # 修改为 log10 表示

    ax.invert_yaxis()
    ax.set_title(f'sr={sr.split("=")[1]}')

    if mappable is None:
        mappable = heatmap.get_children()[0]

# 添加统一 colorbar
cbar_ax = fig.add_axes([1.02, 0.05, 0.02, 0.915])
fig.colorbar(mappable, cax=cbar_ax, label='Average Accuracy (%)')

plt.show()

# %% Topographic map
import numpy as np
import matplotlib.pyplot as plt

# 设置 lambda 和 sigma
lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0])
sigma_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3])

# 构建网格
LAMBDA, SIGMA = np.meshgrid(np.log10(lambda_values), sigma_values, indexing='ij')

# 准备绘图参数
vmin = 60
vmax = 95
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
axes = axes.flatten()
sr_keys = list(accuracy_data.keys())

contour_mappable = None

for i, sr in enumerate(sr_keys):
    ax = axes[i]
    acc_matrix = accuracy_data[sr]  # shape (7, 5)
    
    # 转换为 float 格式数据
    Z = acc_matrix.astype(float)

    # 画等高线填色图（contourf）
    contour = ax.contourf(SIGMA, LAMBDA, Z, levels=20, cmap='coolwarm', vmin=vmin, vmax=vmax)
    
    # 添加等高线线条
    ax.contour(SIGMA, LAMBDA, Z, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    
    # 设置轴标签和刻度
    if i % 3 == 0:
        ax.set_ylabel('log10(lambda)')
    else:
        ax.set_yticklabels([])

    if i >= 3:
        ax.set_xlabel('sigma')
    else:
        ax.set_xticklabels([])

    ax.set_title(f'sr={sr.split("=")[1]}')

    # 可选：标记最大值
    # max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    # ax.plot(SIGMA[max_idx], LAMBDA[max_idx], 'ro')

    # 保存 mappable 用于 colorbar
    if contour_mappable is None:
        contour_mappable = contour

# 添加统一 colorbar（右侧）
cbar_ax = fig.add_axes([1.02, 0.05, 0.02, 0.915])
fig.colorbar(contour_mappable, cax=cbar_ax, label='Average Accuracy (%)')

plt.show()
