# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 23:03:26 2025

@author: 18307
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import networkx as nx
from tqdm import tqdm

# 1. Shannon Entropy
def compute_entropy(fn_matrix):
    flat = fn_matrix.flatten()
    prob = flat / np.sum(flat)
    prob = prob[prob > 0]  # 过滤掉为0的项
    return entropy(prob)

# 2. Mutual Information (pairwise between rows or columns)
def compute_mutual_information(fn_matrix):
    # 使用上三角部分计算MI（去掉对角线）
    triu_indices = np.triu_indices_from(fn_matrix, k=1)
    values = fn_matrix[triu_indices]
    # 离散化为bins
    bins = np.histogram_bin_edges(values, bins='auto')
    digitized = np.digitize(values, bins)
    # 计算每对值之间的互信息（这里示例中近似为自MI）
    return mutual_info_score(digitized, digitized)

# 3. Network Entropy via Graph (based on normalized edge weights)
def compute_graph_entropy(fn_matrix):
    G = nx.from_numpy_array(fn_matrix)
    weights = np.array([d['weight'] for u, v, d in G.edges(data=True)])
    weights = weights[weights > 0]  # 过滤0边权
    prob = weights / weights.sum()
    return entropy(prob)

def example_usage():
    # 模拟数据形状: (samples, N, N)
    # 构造一个例子供测试
    samples = 10
    N = 5
    np.random.seed(42)
    functional_networks = np.abs(np.random.randn(samples, N, N))
    # 强制对称
    functional_networks = 0.5 * (functional_networks + functional_networks.transpose(0, 2, 1))
    
    # 批量计算
    entropy_list = []
    mutual_info_list = []
    graph_entropy_list = []
    
    for i in tqdm(range(samples), desc="Processing"):
        fn = functional_networks[i]
        entropy_list.append(compute_entropy(fn))
        mutual_info_list.append(compute_mutual_information(fn))
        graph_entropy_list.append(compute_graph_entropy(fn))
    
    # 汇总
    results_df = pd.DataFrame({
        'ShannonEntropy': entropy_list,
        'MutualInformation': mutual_info_list,
        'GraphEntropy': graph_entropy_list
    })

if __name__ == '__main__':
    # %% Connectivity Matrix
    from utils import utils_feature_loading
    from spatial_gaussian_smoothing import fcs_residual_filtering
    
    functional_networks_sample = utils_feature_loading.read_fcs_mat(dataset='seed', identifier='sub1ex1', feature='pcc')
    fns_gamma = functional_networks_sample['gamma']
    
    projection_params = {"source": "auto", "type": "3d_spherical"}
    filtering_params = {'sigma': 0.1, 'gamma': 0.1, 'lambda_reg': 0.25}
    
    # origin
    fns = fcs_residual_filtering(fns_gamma, projection_params, 
                                                        residual_type='origin', lateral_mode='bilateral', 
                                                        filtering_params=filtering_params, visualize=True)

    functional_networks = fns
    entropy_list = []
    mutual_info_list = []
    graph_entropy_list = []
    for i in tqdm(range(len(functional_networks)), desc='Processing'):
        fn = functional_networks[i]
        entropy_list.append(compute_entropy(fn))
        mutual_info_list.append(compute_mutual_information(fn))
        graph_entropy_list.append(compute_graph_entropy(fn))
    
    # 汇总
    # 生成 DataFrame 并计算列均值（返回一行 DataFrame）
    results_df = pd.DataFrame({
        'ShannonEntropy': entropy_list,
        'MutualInformation': mutual_info_list,
        'GraphEntropy': graph_entropy_list
    })
    
    # 直接通过 .mean() 和 to_frame().T 实现一行均值表
    results_origin_mean_df = results_df.mean().to_frame().T
    
    # smoothed
    fns_smoothed = fcs_residual_filtering(fns_gamma, projection_params, 
                                                        residual_type='origin_gaussian', lateral_mode='bilateral', 
                                                        filtering_params=filtering_params, visualize=True)
    
    functional_networks = fns_smoothed
    entropy_list = []
    mutual_info_list = []
    graph_entropy_list = []
    for i in tqdm(range(len(functional_networks)), desc='Processing'):
        fn = functional_networks[i]
        entropy_list.append(compute_entropy(fn))
        mutual_info_list.append(compute_mutual_information(fn))
        graph_entropy_list.append(compute_graph_entropy(fn))
    
    # 汇总
    # 生成 DataFrame 并计算列均值（返回一行 DataFrame）
    results_df = pd.DataFrame({
        'ShannonEntropy': entropy_list,
        'MutualInformation': mutual_info_list,
        'GraphEntropy': graph_entropy_list
    })
    
    # 直接通过 .mean() 和 to_frame().T 实现一行均值表
    results_smoothed_mean_df = results_df.mean().to_frame().T
    
    # recovered
    fns_recovered = fcs_residual_filtering(fns_gamma, projection_params, 
                                                        residual_type='pseudoinverse', lateral_mode='bilateral', 
                                                        filtering_params=filtering_params, visualize=True)
    functional_networks = fns_recovered
    entropy_list = []
    mutual_info_list = []
    graph_entropy_list = []
    for i in tqdm(range(len(functional_networks)), desc='Processing'):
        fn = functional_networks[i]
        entropy_list.append(compute_entropy(fn))
        mutual_info_list.append(compute_mutual_information(fn))
        graph_entropy_list.append(compute_graph_entropy(fn))
    
    # 汇总
    # 生成 DataFrame 并计算列均值（返回一行 DataFrame）
    results_df = pd.DataFrame({
        'ShannonEntropy': entropy_list,
        'MutualInformation': mutual_info_list,
        'GraphEntropy': graph_entropy_list
    })
    
    # 直接通过 .mean() 和 to_frame().T 实现一行均值表
    results_recovered_mean_df = results_df.mean().to_frame().T