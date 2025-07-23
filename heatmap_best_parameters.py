# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 17:49:12 2025

@author: usouu
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Matrix Plot
def matrix_plot(accuracy_data, lambda_values, sigma_values):
    log_lambda_labels = np.round(np.log10(lambda_values)).astype(int).astype(str)
    # 准备绘图
    vmin, vmax = 60, 95
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

# %% Topographic plot
def topographic_plot(accuracy_data, lambda_values, sigma_values, max_mark=False):
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
        if max_mark:
            max_idx = np.unravel_index(np.argmax(Z), Z.shape)
            ax.plot(SIGMA[max_idx], LAMBDA[max_idx], 'ro')
    
        # 保存 mappable 用于 colorbar
        if contour_mappable is None:
            contour_mappable = contour

    # 添加统一 colorbar（右侧）
    cbar_ax = fig.add_axes([1.02, 0.05, 0.02, 0.915])
    fig.colorbar(contour_mappable, cax=cbar_ax, label='Average Accuracy (%)')

    plt.show()

# %% Execute
# 原始 lambda 和 sigma
lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0])
sigma_values = [0.05, 0.1, 0.15, 0.2, 0.3]

# accuracy_data 略，保持原样
accuracy_data_pcc = {
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

accuracy_data_plv = {
    "sr=1.0": np.array([
        [73.46378429, 65.63470849, 89.6329899, 90.3462429, 87.33708769],
        [73.96466521, 83.21377636, 91.49291372, 90.02630357, 83.61527349],
        [74.58993013, 89.77056895, 91.25084127, 91.04236196, 87.15292231],
        [74.38260423, 92.38549353, 91.6794955, 91.17806671, 86.44139367],
        [86.70483309, 92.68436578, 92.37171602 ,91.02478972, 85.4144759],
        [93.94103755, 93.18389721, 91.96662601, 89.8066823, 81.96463061],
        [93.2508528, 91.16293394, 87.19949711, 82.80669095, 76.16285608]
    ]),
    "sr=0.5": np.array([
        [71.85485457, 66.70779736, 89.8411578, 90.59404205, 87.21090436],
        [71.48172562, 85.36667849, 91.67162922, 91.00117936, 83.02172741],
        [71.15056359, 90.45682056, 91.93708135, 91.17016583, 86.02437737],
        [73.08506129, 91.65991055, 91.12099009, 91.4906588, 86.58869021],
        [83.01407452, 92.02556539, 91.61065407, 89.88224812, 82.39923067],
        [90.265579, 91.4473943, 90.15567032, 87.53854849, 75.85500451],
        [89.56558448, 88.0319899, 85.01598918, 81.06297344, 74.42134159]
    ]),
    "sr=0.4": np.array([
        [69.54505373, 66.73305132, 89.68180809, 89.91158516, 86.44566129],
        [70.15996101, 83.75259302, 90.35999735, 90.24616707, 81.72103565],
        [69.97370219, 89.54081494, 91.42825342, 90.88308146, 85.82262246],
        [71.12774332, 92.06690946, 90.75918765, 91.62247655, 85.36004637],
        [79.83505624, 91.2547052, 90.72782637, 88.63764104, 82.27741878],
        [90.13239446, 90.588027, 89.76618598, 86.86198554, 73.75838315],
        [88.41118579, 88.28548113, 83.98816022, 78.95209589, 72.20781033]
    ]),
    "sr=0.3": np.array([
        [68.70799344, 66.20952892, 87.71052229, 89.13183793, 85.14687843],
        [68.72658645, 79.11296234, 90.01384672, 90.28427005, 79.17275524],
        [69.82251288, 87.79434078, 89.82328567, 89.97352918 ,86.40613961],
        [70.76053714, 91.52022653, 89.87302658, 90.25289146, 82.56381687],
        [78.51909907, 90.84760249, 89.76026321, 87.66086212, 80.19969608],
        [88.41905207, 89.88320545, 88.75302843, 83.75024582, 72.12181766],
        [86.66004608, 86.46126697, 82.72461988, 76.23089588, 70.01663221]
    ]),
    "sr=0.2": np.array([
        [68.31420687, 64.67615925, 85.16044833, 84.33715978, 80.68092861],
        [67.22874189, 75.42332258, 85.20213266, 87.00695796, 75.01662356],
        [67.73220068, 84.55996447, 87.93570302, 87.04637583, 82.50721315],
        [69.25572597, 88.9556138, 86.00859869, 86.52981427 ,78.31089658],
        [78.00495391, 87.72643939, 86.54975101, 83.81833753, 74.98766714],
        [83.45166769, 86.48096148, 84.24682451, 78.93403345, 68.36754067],
        [80.05596934, 81.3829243, 76.11883609, 71.15735142, 66.96390107]
    ]),
    "sr=0.1": np.array([
        [61.04328469, 59.72604723, 68.58535108, 69.10114563, 69.08968071],
        [60.52947402, 62.89173782, 68.03570965, 74.35020487, 68.32214812],
        [60.94022728, 71.16702855, 73.56023264, 73.19119254, 71.43954532],
        [61.28770433, 75.14143433, 74.74546781, 74.6462455, 69.35864497],
        [66.47994648, 70.30274772, 75.77579968, 73.07962872, 63.66667532],
        [65.36798761, 73.60152481, 69.17522931, 67.35302785, 61.27781959],
        [63.30873104, 65.63031975, 65.23691381, 62.82384219, 56.42387333]
    ])
}

matrix_plot(accuracy_data_pcc, lambda_values, sigma_values)
topographic_plot(accuracy_data_pcc, lambda_values, sigma_values)
topographic_plot(accuracy_data_pcc, lambda_values, sigma_values, max_mark=True)

matrix_plot(accuracy_data_plv, lambda_values, sigma_values)
topographic_plot(accuracy_data_plv, lambda_values, sigma_values)
topographic_plot(accuracy_data_plv, lambda_values, sigma_values, max_mark=True)