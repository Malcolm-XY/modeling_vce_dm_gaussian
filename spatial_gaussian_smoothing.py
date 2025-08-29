# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:21:34 2025

@author: 18307
"""

import numpy as np

import feature_engineering
from utils import utils_feature_loading
from utils import utils_visualization

# %% gaussian filtering
def spatial_gaussian_smoothing_on_vector(A, distance_matrix, sigma):
    dists = distance_matrix
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))
    weights /= weights.sum(axis=1, keepdims=True)
    A_smooth = weights @ A
    return A_smooth

def spatial_gaussian_smoothing_on_fc_matrix(A, distance_matrix, sigma=None, lateral='bilateral', visualize=False):
    """
    Applies spatial Gaussian smoothing to a symmetric functional connectivity (FC) matrix.

    Parameters
    ----------
    A : np.ndarray of shape (N, N)
        Symmetric functional connectivity matrix.
    coordinates : dict with keys 'x', 'y', 'z'
        Each value is a list or array of length N, giving 3D coordinates for each channel.
    sigma : float
        Standard deviation of the spatial Gaussian kernel.
    lateral : str
        'bilateral' or 'unilateral'
    
    Returns
    -------
    A_smooth : np.ndarray of shape (N, N)
        Symmetrically smoothed functional connectivity matrix.
    """
    if visualize:
        try:
            utils_visualization.draw_projection(A, 'Before Spatial Gaussian Smoothing')
        except ModuleNotFoundError: 
            print("utils_visualization not found")
    
    if sigma is None:
        sigma = np.median(distance_matrix[distance_matrix > 0])
    
    # Step 1 & Step 2: Compute Euclidean distance matrix between channels
    dists = distance_matrix

    # Step 3: Compute spatial Gaussian weights
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))  # shape (N, N)
    weights /= weights.sum(axis=1, keepdims=True)       # normalize per row

    # Step 4: Apply spatial smoothing to both rows and columns
    if lateral == 'bilateral':
        A_smooth = weights @ A @ weights.T
    elif lateral == 'unilateral':
        A_smooth = weights @ A

    # Step 5 (optional): Enforce symmetry
    # A_smooth = 0.5 * (A_smooth + A_smooth.T)
    
    if visualize:
        try:
            utils_visualization.draw_projection(A_smooth, 'After Spatial Gaussian Smoothing')
        except ModuleNotFoundError: 
            print("utils_visualization not found")
    
    return A_smooth

def fcs_gaussian_filtering(fcs, projection_params={"source": "auto", "type": "3d_spherical"}, lateral='bilateral', sigma=0.05):
    """
    projection_params:
        "source": "auto", or "manual"
        "type": "2d", "3d", or "stereo"
    """
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed', projection_params=projection_params)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)
    
    fcs_temp = []
    for fc in fcs:
        fcs_temp.append(spatial_gaussian_smoothing_on_fc_matrix(fc, distance_matrix, sigma))

    fcs = np.stack(fcs_temp)
    
    return fcs

def cfs_gaussian_filtering(cfs, projection_params={"source": "auto", "type": "3d_spherical"}, lateral='bilateral', sigma=0.05):
    """
    projection_params:
        "source": "auto", or "manual"
        "type": "2d", "3d", or "stereo"
    """
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed', projection_params=projection_params)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)
    
    cfs_temp = []
    for cf in cfs:
        cfs_temp.append(spatial_gaussian_smoothing_on_vector(cf, distance_matrix, sigma))

    cfs = np.stack(cfs_temp)
    
    return cfs

# %% apply filters
def apply_gaussian_filter(matrix, distance_matrix, 
                          filtering_params={'computation': 'gaussian_filter',
                                            'sigma': 0.1, 
                                            'lateral_mode': 'bilateral', 'reinforce': False}, 
                          visualize=False):
    """
    Applies a spatial residual filter to a functional connectivity (FC) matrix
    to suppress local spatial redundancy (e.g., volume conduction effects).

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Input functional connectivity matrix.
    distance_matrix : np.ndarray, shape (N, N)
        Pairwise distance matrix between channels.
    params : dict
        Filtering parameters:
            - 'sigma': float, spatial Gaussian kernel width. If None, uses mean of non-zero distances.
    lateral_mode : str
        'bilateral' (K @ M @ K.T) or 'unilateral' (K @ M).
    visualize : bool
        If True, visualize before and after matrices.

    Returns
    -------
    filtered_matrix : np.ndarray
        Filtered connectivity matrix.
    """

    if visualize:
        try:
            utils_visualization.draw_projection(matrix, 'Before Spatial Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    lateral_mode = filtering_params.get('lateral_mode', 'bilateral')
    sigma = filtering_params.get('sigma', 0.1)

    # Avoid zero distances
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)

    # Step 1: Construct Gaussian kernel (SM)
    gaussian_kernel = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum(axis=1, keepdims=True)

    # Step 2: Apply filtering
    if lateral_mode == 'bilateral':
        filtered_matrix = gaussian_kernel @ matrix @ gaussian_kernel.T
    elif lateral_mode == 'unilateral':
        filtered_matrix = gaussian_kernel @ matrix
    else:
        raise ValueError(f"Unknown lateral_mode: {lateral_mode}")
    
    # Step 4: Reinforce
    if filtering_params.get('reinforce', False):
        filtered_matrix += matrix
    
    if visualize:
        try:
            utils_visualization.draw_projection(filtered_matrix, 'After Spatial Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return filtered_matrix

# gaussian diffusion inverse filtering: proposed method
def apply_diffusion_inverse(matrix, distance_matrix, 
                            filtering_params={'computation': 'diffusion_inverse',
                                              'sigma': 0.1, 'lambda_reg': 0.01,
                                              'lateral_mode': 'bilateral', 'reinforce': False}, 
                            visualize=False):
    """
    Applies a spatial residual filter to a functional connectivity (FC) matrix
    to suppress local spatial redundancy (e.g., volume conduction effects).

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Input functional connectivity matrix.
    distance_matrix : np.ndarray, shape (N, N)
        Pairwise distance matrix between channels.
    params : dict
        Filtering parameters:
            - 'sigma': float, spatial Gaussian kernel width. If None, uses mean of non-zero distances.
            - 'lambda_reg': float, regularization term for pseudoinverse mode.
    lateral_mode : str
        'bilateral' (K @ M @ K.T) or 'unilateral' (K @ M).
    visualize : bool
        If True, visualize before and after matrices.

    Returns
    -------
    filtered_matrix : np.ndarray
        Filtered connectivity matrix.
    """

    if visualize:
        try:
            utils_visualization.draw_projection(matrix, 'Before Spatial Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    lateral_mode = filtering_params.get('lateral_mode', 'bilateral')
    sigma = filtering_params.get('sigma', 0.1)
    lambda_reg = filtering_params.get('lambda_reg', 0.01)

    # Avoid zero distances
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)

    # Step 1: Construct Gaussian kernel (SM)
    gaussian_kernel = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum(axis=1, keepdims=True)

    # Step 2: Construct residual kernel
    # Use Tikhonov regularized inverse
    I = np.eye(gaussian_kernel.shape[0])
    G = gaussian_kernel
    try:
        residual_kernel = np.linalg.inv(G.T @ G + lambda_reg * I) @ G.T
    except np.linalg.LinAlgError:
        print('LinAlgError')
        residual_kernel = np.linalg.pinv(G)

    # Step 3: Apply filtering
    if lateral_mode == 'bilateral':
        filtered_matrix = residual_kernel @ matrix @ residual_kernel.T
    elif lateral_mode == 'unilateral':
        filtered_matrix = residual_kernel @ matrix
    else:
        raise ValueError(f"Unknown lateral_mode: {lateral_mode}")
    
    # Step 4: Reinforce
    if filtering_params.get('reinforce', False):
        filtered_matrix += matrix
    
    if visualize:
        try:
            utils_visualization.draw_projection(filtered_matrix, 'After Spatial Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return filtered_matrix

# laplacian graph filtering
def apply_graph_laplacian_filtering(matrix, distance_matrix,
                                 filtering_params={'computation': 'graph_laplacian_filtering',
                                                   'alpha': 0.1,
                                                   'sigma': None,  # 新增
                                                   'lateral_mode': 'bilateral',
                                                   'normalized': False,
                                                   'reinforce': False},
                                 visualize=False):
    """
    ...
    sigma : float or None
        高斯核的尺度参数。如果为 None，则默认取非零距离的均值。
    """

    alpha = filtering_params.get('alpha', 0.1)
    sigma = filtering_params.get('sigma', None)  # 取出用户自定义 sigma
    lateral_mode = filtering_params.get('lateral_mode', 'bilateral')
    normalized = filtering_params.get('normalized', False)
    reinforce = filtering_params.get('reinforce', False)

    # Step 1: Construct adjacency matrix W (Gaussian kernel)
    if sigma is None:
        sigma = np.mean(distance_matrix[distance_matrix > 0])
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)
    W = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))

    # Step 2: Compute Laplacian matrix L
    D = np.diag(W.sum(axis=1))
    if normalized:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W

    # Step 3: Construct filter matrix F = I - alpha * L
    I = np.eye(W.shape[0])
    F = I - alpha * L

    # Step 4: Apply filtering
    if lateral_mode == 'bilateral':
        filtered_matrix = F @ matrix @ F.T
    elif lateral_mode == 'unilateral':
        filtered_matrix = F @ matrix
    else:
        raise ValueError(f"Unknown lateral_mode: {lateral_mode}")

    # Step 5: Optional reinforcement
    if reinforce:
        filtered_matrix += matrix

    return filtered_matrix

# graph spectral filtering
def apply_graph_spectral_filtering(matrix, distance_matrix,
                                   filtering_params={'computation': 'graph_spectral_filtering',
                                                     'cutoff_rank': 5,
                                                     'normalized': False, 'reinforce': False},
                                   visualize=False):
    """
    Applies Graph Laplacian Denoising to a functional connectivity matrix by removing
    low-frequency components in the graph spectral domain (i.e., spatially smooth structures).

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Input functional connectivity matrix (e.g., PCC or PLV).
    distance_matrix : np.ndarray, shape (N, N)
        Pairwise spatial distances between EEG channels.
    cutoff_rank : int
        Number of lowest-frequency graph modes to remove (i.e., spectral truncation level).
    normalized : bool
        If True, use normalized Laplacian; otherwise use unnormalized Laplacian.
    reinforce : bool
        If True, adds original matrix back to filtered result (residual enhancement).
    visualize : bool
        If True, show pre- and post-filter visualization.

    Returns
    -------
    filtered_matrix : np.ndarray
        The denoised functional connectivity matrix.
    """
    cutoff_rank = filtering_params.get('cutoff_rank', 5)
    normalized = filtering_params.get('normalized', False)
    reinforce = filtering_params.get('reinforce', False)

    if visualize:
        try:
            utils_visualization.draw_projection(matrix, 'Before Graph Spectral Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    # Step 1: Construct adjacency matrix W (Gaussian kernel)
    sigma = np.mean(distance_matrix[distance_matrix > 0])
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)
    W = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))

    # Step 2: Compute Laplacian matrix L
    D = np.diag(W.sum(axis=1))
    if normalized:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W

    # Step 3: Eigen-decomposition of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)  # L symmetric => use eigh
    # Sort eigenvalues/eigenvectors from low to high freq
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    # Step 4: Construct projection matrix to remove low frequencies
    U_low = eigenvectors[:, :cutoff_rank]  # Low-frequency basis
    P_low = U_low @ U_low.T               # Projector onto low-frequency subspace

    # Step 5: Filter matrix by removing smooth components
    # filtered_matrix = (np.eye(matrix.shape[0]) - P_low) @ matrix @ (np.eye(matrix.shape[0]) - P_low.T) # High Pass
    
    filtered_matrix = P_low @ matrix @ P_low.T # Low Pass

    # Step 6: Optional residual reinforcement
    if reinforce:
        filtered_matrix += matrix

    if visualize:
        try:
            utils_visualization.draw_projection(filtered_matrix, 'After Graph Laplacian Denoising')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return filtered_matrix

# graph tikhonov inverse
from scipy.sparse.linalg import cg
from scipy.sparse import identity, kron, csc_matrix
def apply_graph_tikhonov_inverse(matrix, distance_matrix,
                                 filtering_params={'computation': 'graph_tikhonov_inverse',
                                                   'alpha': 0.1, 'lambda': 1e-2,
                                                   'normalized': False, 'reinforce': False},
                                 visualize=False):
    """
    Applies Graph Tikhonov Inverse Filtering to estimate the true functional connectivity matrix
    from an observed matrix, based on a diffusion model and smoothness regularization.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Input observed functional connectivity matrix (FN_obs).
    distance_matrix : np.ndarray, shape (N, N)
        Pairwise spatial distances between EEG channels.
    alpha : float
        Diffusion parameter in filter matrix G = I - alpha * L.
    lambda : float
        Regularization strength (controls how much smoothness is penalized).
    normalized : bool
        Whether to use normalized Laplacian.
    reinforce : bool
        If True, adds original matrix back to filtered result.
    visualize : bool
        If True, show pre- and post-filter visualization.

    Returns
    -------
    estimated_matrix : np.ndarray
        Estimated FN_true after Tikhonov-regularized inverse filtering.
    """
    alpha = filtering_params.get('alpha', 0.1)
    lam = filtering_params.get('lambda', 1e-2)
    normalized = filtering_params.get('normalized', False)
    reinforce = filtering_params.get('reinforce', False)

    # Step 1: 构造高斯核权重矩阵 W
    sigma = np.mean(distance_matrix[distance_matrix > 0])
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)
    W = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))

    # Step 2: 计算拉普拉斯矩阵 L
    D = np.diag(W.sum(axis=1))
    if normalized:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W

    # Step 3: 构造滤波矩阵 G = I - alpha * L
    I = np.eye(W.shape[0])
    G = I - alpha * L

    # 若需要可视化
    if visualize:
        try:
            utils_visualization.draw_projection(matrix, 'Before Graph Tikhonov Inverse (Simplified)')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    # 构造右侧项 M_G = G^T M G
    M_G = G.T @ matrix @ G

    # 计算 A = Kron(GTG, GTG) + lambda * I
    GTG = G.T @ G
    N = GTG.shape[0]
    # 注意：这里使用稀疏矩阵以节省计算资源
    GTG_sparse = csc_matrix(GTG)
    A_kron = kron(GTG_sparse, GTG_sparse)
    I_big = identity(N * N, format='csc')
    A = A_kron + lam * I_big

    # 构造右侧向量 b = vec(M_G)
    b = M_G.flatten()

    # 使用共轭梯度法求解稀疏线性系统
    x, info = cg(A, b, rtol=1e-6, maxiter=1000)
    if info != 0:
        print("共轭梯度法未能完全收敛：info =", info)

    # 重构回矩阵 X
    estimated_matrix = x.reshape(matrix.shape)

    if reinforce:
        estimated_matrix += matrix

    if visualize:
        try:
            utils_visualization.draw_projection(estimated_matrix, 'After Graph Tikhonov Inverse (Simplified)')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return estimated_matrix

# spectral graph filtering
def apply_exp_graph_spectral_filtering(matrix, distance_matrix,
                                   filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                     't': 10, 
                                                     'normalized': False, 'reinforce': False},
                                   visualize=False):
    """
    Applies Exp Graph Spectral Filtering (Low-pass) to a functional connectivity matrix
    by attenuating low-frequency components in the graph Laplacian spectrum.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Input functional connectivity matrix (e.g., PCC or PLV).
    distance_matrix : np.ndarray, shape (N, N)
        Pairwise spatial distances between EEG channels.
    t : float
        Controls the strength of filtering; larger t suppresses low frequencies more.
    normalized : bool
        If True, use normalized Laplacian; otherwise unnormalized.
    reinforce : bool
        If True, adds original matrix back to filtered result.
    visualize : bool
        If True, show pre- and post-filter visualization.

    Returns
    -------
    filtered_matrix : np.ndarray
        The spectrally filtered functional connectivity matrix.
    """
    t = filtering_params.get('t', 10)
    normalized = filtering_params.get('normalized', False)
    reinforce = filtering_params.get('reinforce', False)

    if visualize:
        try:
            utils_visualization.draw_projection(matrix, 'Before Exp Graph Spectral Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    # Step 1: Construct adjacency matrix W (Gaussian kernel)
    sigma = np.mean(distance_matrix[distance_matrix > 0])
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)
    W = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))

    # Step 2: Compute Laplacian matrix L
    D = np.diag(W.sum(axis=1))
    if normalized:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W

    # Step 3: Spectral decomposition of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 4: Construct high-pass filter h(λ) = 1 - exp(-t * λ)
    h_lambda = np.exp(-t * eigenvalues)
    H = eigenvectors @ np.diag(h_lambda) @ eigenvectors.T

    # Step 5: Filter matrix: apply H on both sides
    filtered_matrix = H @ matrix @ H.T

    # Step 6: Optional residual reinforcement
    if reinforce:
        filtered_matrix += matrix

    if visualize:
        try:
            utils_visualization.draw_projection(filtered_matrix, 'After Exp Graph Spectral Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return filtered_matrix

# %% aplly filters on fcs
def fcs_filtering_common(fcs,
                         projection_params={"source": "auto", "type": "3d_spherical"},
                         filtering_params={}, 
                         apply_filter='diffusion_inverse',
                         visualize=False):
    # valid filters
    filters_valid={'gaussian_filtering', 
                   'diffusion_inverse', 
                   'graph_laplacian', 'graph_spectral_filtering',
                   'graph_tikhonov_inverse', 'exp_graph_spectral_filtering'}
    
    # default parameteres
    filtering_params_gaussian_filtering={'computation': 'gaussian_filtering',
                                         'sigma': 0.1, 
                                         'lateral_mode': 'bilateral', 'reinforce': False}
    
    filtering_params_diffusion_inverse={'computation': 'diffusion_inverse',
                                        'sigma': 0.1, 'lambda_reg': 0.01,
                                        'lateral_mode': 'bilateral', 'reinforce': False}
    
    filtering_params_graph_laplacian={'computation': 'graph_laplacian_filtering',
                                      'alpha': 0.1, 'sigma': None,
                                      'lateral_mode': 'bilateral', 'normalized': False, 'reinforce': False}
    
    filtering_params_graph_laplacian_denoising={'computation': 'graph_spectral_filtering',
                                                'cutoff_rank': 5,
                                                'normalized': False, 'reinforce': False}
    
    filtering_params_graph_tikhonov_inverse={'computation': 'graph_tikhonov_inverse',
                                             'alpha': 0.1, 'lambda': 1e-2,
                                             'normalized': False, 'reinforce': False}
    
    filtering_params_exp_graph_spectral_filtering={'computation': 'exp_graph_spectral_filtering', 
                      't': 10, 
                      'normalized': False, 'reinforce': False}
    
    # Step 1: Compute spatial distance matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed', projection_params=projection_params)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    # Step 2: Apply filtering to each FC matrix
    if fcs.ndim == 2:
        if apply_filter=='gaussian_filtering':
            fcs_filtered = apply_gaussian_filter(matrix=fcs, distance_matrix=distance_matrix, 
                                                 filtering_params=filtering_params,
                                                 visualize=False)
            
        elif apply_filter=='diffusion_inverse':
            fcs_filtered = apply_diffusion_inverse(matrix=fcs, distance_matrix=distance_matrix, 
                                                   filtering_params=filtering_params,
                                                   visualize=False)
        elif apply_filter=='graph_laplacian_filtering':
            fcs_filtered=apply_graph_laplacian_filtering(matrix=fcs, distance_matrix=distance_matrix, 
                                                      filtering_params=filtering_params,
                                                      visualize=False)
        elif apply_filter=='graph_spectral_filtering':
            fcs_filtered=apply_graph_spectral_filtering(matrix=fcs, distance_matrix=distance_matrix, 
                                                        filtering_params=filtering_params,
                                                        visualize=False)
        elif apply_filter=='graph_tikhonov_inverse':
            fcs_filtered=apply_graph_tikhonov_inverse(matrix=fcs, distance_matrix=distance_matrix, 
                                                      filtering_params=filtering_params,
                                                      visualize=False)
        elif apply_filter=='exp_graph_spectral_filtering':
            fcs_filtered=apply_exp_graph_spectral_filtering(matrix=fcs, distance_matrix=distance_matrix, 
                                                            filtering_params=filtering_params,
                                                            visualize=False)
        
        if visualize:
            utils_visualization.draw_projection(fcs_filtered)
        
    elif fcs.ndim == 3:
        fcs_filtered = []
        if apply_filter=='gaussian_filtering':
            for fc in fcs:
                filtered = apply_gaussian_filter(matrix=fc, distance_matrix=distance_matrix, 
                                                 filtering_params=filtering_params,
                                                 visualize=False)
                fcs_filtered.append(filtered)
                
        elif apply_filter=='diffusion_inverse':
            for fc in fcs:
                filtered = apply_diffusion_inverse(matrix=fc, distance_matrix=distance_matrix, 
                                                   filtering_params=filtering_params,
                                                   visualize=False)
                fcs_filtered.append(filtered)
                
        elif apply_filter=='graph_laplacian_filtering':
            for fc in fcs:
                filtered = apply_graph_laplacian_filtering(matrix=fc, distance_matrix=distance_matrix, 
                                                        filtering_params=filtering_params,
                                                        visualize=False)
                fcs_filtered.append(filtered)
                
        elif apply_filter=='graph_spectral_filtering':
            for fc in fcs:
                filtered = apply_graph_spectral_filtering(matrix=fc, distance_matrix=distance_matrix, 
                                                          filtering_params=filtering_params,
                                                          visualize=False)
                fcs_filtered.append(filtered)
                
        elif apply_filter=='graph_tikhonov_inverse':
            for fc in fcs:
                filtered = apply_graph_tikhonov_inverse(matrix=fc, distance_matrix=distance_matrix, 
                                                        filtering_params=filtering_params,
                                                        visualize=False)
                fcs_filtered.append(filtered)
        
        elif apply_filter=='exp_graph_spectral_filtering':
            for fc in fcs:
                filtered = apply_exp_graph_spectral_filtering(matrix=fc, distance_matrix=distance_matrix, 
                                                              filtering_params=filtering_params,
                                                              visualize=False)
                fcs_filtered.append(filtered)
        
        if visualize:
            average = np.mean(fcs_filtered, axis=0)
            utils_visualization.draw_projection(average)
        
    return np.stack(fcs_filtered)

# %% Usage
if __name__ == '__main__':
    # electrodes = utils_feature_loading.read_distribution('seed')['channel']
    
    # %% Distance Matrix
    _, distance_matrix_euc = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                        projection_params={"source": "auto", "type": "3d_euclidean"})
    distance_matrix_euc = feature_engineering.normalize_matrix(distance_matrix_euc)
    utils_visualization.draw_projection(distance_matrix_euc) # , xticklabels=electrodes, yticklabels=electrodes)
    
    _, distance_matrix_sph = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                        projection_params={"source": "auto", "type": "3d_spherical"})
    distance_matrix_sph = feature_engineering.normalize_matrix(distance_matrix_sph)
    utils_visualization.draw_projection(distance_matrix_sph) # , xticklabels=electrodes, yticklabels=electrodes)
    
    sigma = 0.1
    gaussian_kernel = np.exp(-np.square(distance_matrix_euc) / (2 * sigma ** 2))
    utils_visualization.draw_projection(gaussian_kernel)

    gaussian_kernel = np.exp(-np.square(distance_matrix_sph) / (2 * sigma ** 2))
    utils_visualization.draw_projection(gaussian_kernel)
    
    # %% Connectivity Matrix
    # get sample and visualize sample
    sample_averaged = utils_feature_loading.read_fcs_global_average('seed', 'pcc', 'gamma', sub_range=range(1, 16))
    utils_visualization.draw_projection(sample_averaged)
    
    # gaussian filtering; gaussian; sigma = 0.1
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'gaussian_filtering',
                                                         'sigma': 0.1,
                                                         'lateral_mode': 'bilateral', 'reinforce': False}, 
                                       apply_filter='gaussian_filtering',
                                       visualize=True)
    
    # gaussian diffusion inverse; sigma = 0.1, lambda = 0.1
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'diffusion_inverse',
                                                         'sigma': 0.1, 'lambda_reg': 0.1,
                                                         'lateral_mode': 'bilateral', 'reinforce': False}, 
                                       apply_filter='diffusion_inverse',
                                       visualize=True)
    
    # gaussian diffusion inverse; sigma = 0.1, lambda = 0.01
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'diffusion_inverse',
                                                         'sigma': 0.1, 'lambda_reg': 0.01,
                                                         'lateral_mode': 'bilateral', 'reinforce': False}, 
                                       apply_filter='diffusion_inverse',
                                       visualize=True)
    
    # graph_laplacian_filtering; alpha = 0.1
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'graph_laplacian_filtering',
                                                         'alpha': 0.1, 'sigma': None,
                                                         'lateral_mode': 'bilateral', 'normalized': False, 'reinforce': False}, 
                                       apply_filter='graph_laplacian_filtering',
                                       visualize=True)
    
    # graph_spectral_filtering; cutoff rank = 5
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'graph_spectral_filtering',
                                                         'cutoff_rank': 5,
                                                         'normalized': False, 'reinforce': False}, 
                                       apply_filter='graph_spectral_filtering',
                                       visualize=True)
    
    # graph_tikhonov_inverse; alpha = 0.1, lambda = 1e-2
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'graph_tikhonov_inverse',
                                                         'alpha': 0.1, 'lambda': 1e-2,
                                                         'normalized': False, 'reinforce': False}, 
                                       apply_filter='graph_tikhonov_inverse',
                                       visualize=True)
    
    # graph_tikhonov_inverse; alpha = 0.1, lambda = 1e-2
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                         't': 10, 
                                                         'normalized': False, 'reinforce': False}, 
                                       apply_filter='exp_graph_spectral_filtering',
                                       visualize=True)
    
    # %% Channel Feature
    # de_sample = utils_feature_loading.read_cfs('seed', 'sub1ex1', 'de_LDS')
    # gamma = de_sample['gamma']
    
    # de_gamma_average = np.mean(gamma, axis=0)
    # utils_visualization.draw_heatmap_1d(de_gamma_average)
    
    # de_gamma_smoothed_sample = cfs_gaussian_filtering(gamma)
    # de_gamma_smoothed_average = np.mean(de_gamma_smoothed_sample, axis=0)
    # utils_visualization.draw_heatmap_1d(de_gamma_smoothed_average)