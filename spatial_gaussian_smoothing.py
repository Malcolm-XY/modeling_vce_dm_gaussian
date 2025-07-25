# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:21:34 2025

@author: 18307
"""

import numpy as np

import feature_engineering
from utils import utils_feature_loading
from utils import utils_visualization

from scipy.spatial.distance import cdist
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

def fcs_gaussian_filtering(fcs, projection_params={"source": "auto", "type": "3d"}, lateral='bilateral', sigma=0.05):
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

def cfs_gaussian_filtering(cfs, projection_params={"source": "auto", "type": "3d"}, lateral='bilateral', sigma=0.05):
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

# %% spatial filtering
def apply_spatial_filter(matrix, distance_matrix, 
                         filter_params={'computation': 'gaussian', 'lateral_mode': 'bilateral',
                                 'sigma': None, 'lambda_reg': None, 'reinforce': False}, 
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
            - 'gamma': float, residual scaling for 'residual' mode. Default 0.25.
            - 'lambda_reg': float, regularization term for pseudoinverse mode.
    computation : str
        One of ['origin', 'residual', 'residual_mean', 'inverse', 'pseudoinverse'].
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
            
    computation = filter_params.get('computation', 'gaussian')
    lateral_mode = filter_params.get('lateral_mode', 'bilateral')
    sigma = filter_params.get('sigma', 0.1)
    lambda_reg = filter_params.get('lambda_reg', 0.01)

    if sigma is None:
        sigma = np.mean(distance_matrix[distance_matrix > 0])

    # Avoid zero distances
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)

    # Step 1: Construct Gaussian kernel (SM)
    gaussian_kernel = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum(axis=1, keepdims=True)

    # Step 2: Construct residual kernel
    if computation == 'origin':
        return matrix
    
    elif computation == 'gaussian':
        residual_kernel = gaussian_kernel

    elif computation == 'pseudoinverse':
        # Use Tikhonov regularized inverse
        I = np.eye(gaussian_kernel.shape[0])
        G = gaussian_kernel
        try:
            residual_kernel = np.linalg.inv(G.T @ G + lambda_reg * I) @ G.T
        except np.linalg.LinAlgError:
            print('LinAlgError')
            residual_kernel = np.linalg.pinv(G)

    elif computation == 'wiener':
        # Construct Wiener-like inverse filter: G.T @ (G G.T + λI)^(-1)
        G = gaussian_kernel
        I = np.eye(G.shape[0])
        try:
            inverse_term = np.linalg.inv(G @ G.T + lambda_reg * I)
        except np.linalg.LinAlgError:
            print("Warning: Matrix inversion failed, falling back to pinv.")
            inverse_term = np.linalg.pinv(G @ G.T + lambda_reg * I)
        residual_kernel = G.T @ inverse_term  # Shape: (N, N)

    else:
        raise ValueError(f"Unknown computation: {computation}")

    # Step 3: Apply filtering
    if lateral_mode == 'bilateral':
        filtered_matrix = residual_kernel @ matrix @ residual_kernel.T
    elif lateral_mode == 'unilateral':
        filtered_matrix = residual_kernel @ matrix
    else:
        raise ValueError(f"Unknown lateral_mode: {lateral_mode}")
    
    # Step 4: Reinforce
    if filter_params.get('reinforce', False):
        filtered_matrix += matrix
    
    if visualize:
        try:
            utils_visualization.draw_projection(filtered_matrix, 'After Spatial Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return filtered_matrix

def fcs_spatial_filter(fcs,
                       projection_params={"source": "auto", "type": "3d_euclidean"},
                       filter_params={'computation': 'gaussian', 'lateral_mode': 'bilateral',
                                      'sigma': None, 'lambda_reg': None, 'reinforce': False},
                       visualize=False):
    """
    Applies spatial residual filtering to a list/array of functional connectivity matrices (FCs),
    using a Gaussian-based distance kernel and selected residual strategy.

    Parameters
    ----------
    fcs : list or np.ndarray of shape (N, C, C)
        A list or array of N functional connectivity matrices (each C x C).

    projection_params : dict
        Parameters for computing the spatial projection (distance matrix), passed to `compute_distance_matrix`.
            - "source": "auto" or "manual"
            - "type": "2d", "3d", or "stereo"

    filtering_params : dict
        Parameters for Gaussian filtering and inverse/residual construction.
            - 'sigma': float or None, spatial kernel width (if None, uses mean non-zero distance)
            - 'gamma': float, strength for residual subtraction (used in 'residual')
            - 'lambda_reg': float, regularization term for pseudoinverse/Wiener filters

    residual_type : str
        One of ['origin', 'residual', 'residual_mean', 'inverse', 'pseudoinverse', 'wiener', 'laplacian_power'].
        Determines how the spatial kernel is transformed into a residual/inverse form.

    lateral_mode : str
        Either 'bilateral' (K @ M @ K.T) or 'unilateral' (K @ M). Controls how kernel is applied.

    visualize : bool
        Whether to visualize each FC matrix before and after filtering.

    Returns
    -------
    fcs_filtered : np.ndarray of shape (N, C, C)
        Stack of filtered FC matrices.
    """
    # Step 1: Compute spatial distance matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed', projection_params=projection_params)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    # Step 2: Apply filtering to each FC matrix    
    fcs_filtered = []
    for fc in fcs:
        filtered = apply_spatial_filter(matrix=fc, distance_matrix=distance_matrix, 
                                 filter_params=filter_params, 
                                 visualize=False)

        fcs_filtered.append(filtered)
    
    if visualize:
        average = np.mean(fcs_filtered, axis=0)
        utils_visualization.draw_projection(average)
        
    return np.stack(fcs_filtered)

# %% -------------spare codes
def apply_spatial_residual_filter_(matrix, distance_matrix, 
                                  residual_type='residual', lateral_mode='bilateral', 
                                  params={'sigma': None, 'gamma': None, 'lambda_reg': None, 'reinforce': False}, 
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
            - 'gamma': float, residual scaling for 'residual' mode. Default 0.25.
            - 'lambda_reg': float, regularization term for pseudoinverse mode.
    residual_type : str
        One of ['origin', 'residual', 'residual_mean', 'inverse', 'pseudoinverse'].
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

    sigma = params.get('sigma', 0.1)
    gamma = params.get('gamma', 0.1)
    lambda_reg = params.get('lambda_reg', 0.1)

    if sigma is None:
        sigma = np.mean(distance_matrix[distance_matrix > 0])

    # Avoid zero distances
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)

    # Step 1: Construct Gaussian kernel (SM)
    gaussian_kernel = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum(axis=1, keepdims=True)

    # Step 2: Construct residual kernel
    if residual_type == 'origin':
        return matrix
    
    elif residual_type == 'gaussian':
        residual_kernel = gaussian_kernel
    
    elif residual_type == 'residual':
        residual_kernel = 1.0 - gamma * gaussian_kernel
        residual_kernel = np.maximum(residual_kernel, 0)
        residual_kernel /= residual_kernel.sum(axis=1, keepdims=True)

    elif residual_type == 'residual_mean':
        row_mean = np.mean(gaussian_kernel, axis=1, keepdims=True)
        residual_kernel = -(gaussian_kernel - row_mean) + row_mean
        residual_kernel /= residual_kernel.sum(axis=1, keepdims=True)

    elif residual_type == 'inverse':
        residual_kernel = 1.0 / (gaussian_kernel + 1e-6)
        residual_kernel /= residual_kernel.sum(axis=1, keepdims=True)

    elif residual_type == 'pseudoinverse':
        # Use Tikhonov regularized inverse
        I = np.eye(gaussian_kernel.shape[0])
        G = gaussian_kernel
        try:
            residual_kernel = np.linalg.inv(G.T @ G + lambda_reg * I) @ G.T
        except np.linalg.LinAlgError:
            print('LinAlgError')
            residual_kernel = np.linalg.pinv(G)

    elif residual_type == 'wiener':
        # Construct Wiener-like inverse filter: G.T @ (G G.T + λI)^(-1)
        G = gaussian_kernel
        I = np.eye(G.shape[0])
        try:
            inverse_term = np.linalg.inv(G @ G.T + lambda_reg * I)
        except np.linalg.LinAlgError:
            print("Warning: Matrix inversion failed, falling back to pinv.")
            inverse_term = np.linalg.pinv(G @ G.T + lambda_reg * I)
        residual_kernel = G.T @ inverse_term  # Shape: (N, N)

    else:
        raise ValueError(f"Unknown residual_type: {residual_type}")

    # Step 3: Apply filtering
    if lateral_mode == 'bilateral':
        filtered_matrix = residual_kernel @ matrix @ residual_kernel.T
    elif lateral_mode == 'unilateral':
        filtered_matrix = residual_kernel @ matrix
    else:
        raise ValueError(f"Unknown lateral_mode: {lateral_mode}")
    
    # Step 4: Reinforce
    if params.get('reinforce', False):
        filtered_matrix += matrix
    
    if visualize:
        try:
            utils_visualization.draw_projection(filtered_matrix, 'After Spatial Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return filtered_matrix

def fcs_residual_filtering_(fcs,
                           projection_params={"source": "auto", "type": "3d_spherical"},
                           residual_type='residual', lateral_mode='bilateral',
                           filtering_params={'sigma': None, 'gamma': None, 'lambda_reg': None, 'reinforce': False}, 
                           visualize=False):
    """
    Applies spatial residual filtering to a list/array of functional connectivity matrices (FCs),
    using a Gaussian-based distance kernel and selected residual strategy.

    Parameters
    ----------
    fcs : list or np.ndarray of shape (N, C, C)
        A list or array of N functional connectivity matrices (each C x C).

    projection_params : dict
        Parameters for computing the spatial projection (distance matrix), passed to `compute_distance_matrix`.
            - "source": "auto" or "manual"
            - "type": "2d", "3d", or "stereo"

    filtering_params : dict
        Parameters for Gaussian filtering and inverse/residual construction.
            - 'sigma': float or None, spatial kernel width (if None, uses mean non-zero distance)
            - 'gamma': float, strength for residual subtraction (used in 'residual')
            - 'lambda_reg': float, regularization term for pseudoinverse/Wiener filters

    residual_type : str
        One of ['origin', 'residual', 'residual_mean', 'inverse', 'pseudoinverse', 'wiener', 'laplacian_power'].
        Determines how the spatial kernel is transformed into a residual/inverse form.

    lateral_mode : str
        Either 'bilateral' (K @ M @ K.T) or 'unilateral' (K @ M). Controls how kernel is applied.

    visualize : bool
        Whether to visualize each FC matrix before and after filtering.

    Returns
    -------
    fcs_filtered : np.ndarray of shape (N, C, C)
        Stack of filtered FC matrices.
    """
    # Step 1: Compute spatial distance matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed', projection_params=projection_params)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    # Step 2: Apply filtering to each FC matrix
    fcs_filtered = []
    for fc in fcs:
        filtered = apply_spatial_residual_filter(matrix=fc, distance_matrix=distance_matrix,
                                                 residual_type=residual_type,lateral_mode=lateral_mode,
                                                 params=filtering_params)
        fcs_filtered.append(filtered)
    
    if visualize:
        average = np.mean(fcs_filtered, axis=0)
        utils_visualization.draw_projection(average)
        
    return np.stack(fcs_filtered)

# %% laplacian graph filtering
def apply_laplacian_filter(matrix, distance_matrix, 
                           filtering_params={'alpha': 0.1, 'lateral_mode': 'bilateral', 'normalized': False, 'reinforce': False},
                           visualize=False):
    """
    Applies a Laplacian-based spatial filter to a functional connectivity matrix to suppress
    volume conduction effects by enhancing spatial differences (high-pass filtering).

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Input functional connectivity matrix (e.g., PCC or PLV).
    distance_matrix : np.ndarray, shape (N, N)
        Pairwise spatial distances between EEG channels.
    alpha : float
        Filtering strength (0 < alpha < 1). Controls how much Laplacian is applied.
    normalized : bool
        If True, use normalized Laplacian; otherwise use unnormalized Laplacian.
    lateral_mode : str
        'bilateral' (L @ M @ L.T) or 'unilateral' (L @ M).
    reinforce : bool
        If True, adds original matrix back to filtered result (residual enhancement).
    visualize : bool
        If True, show pre- and post-filter visualization.

    Returns
    -------
    filtered_matrix : np.ndarray
        The Laplacian-filtered functional connectivity matrix.
    """
    alpha = filtering_params.get('alpha', 0.1)
    lateral_mode = filtering_params.get('lateral_mode', 'bilateral')
    normalized = filtering_params.get('normalized', False)
    reinforce = filtering_params.get('reinforce', False)

    if visualize:
        try:
            utils_visualization.draw_projection(matrix, 'Before Laplacian Filtering')
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

    if visualize:
        try:
            utils_visualization.draw_projection(filtered_matrix, 'After Laplacian Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return filtered_matrix

def fcs_laplacian_geaph_filtering(fcs,
                           projection_params={"source": "auto", "type": "3d_euclidean"},
                           filtering_params={'alpha': 0.1, 'lateral_mode': 'bilateral', 'normalized': False, 'reinforce': False}, 
                           visualize=False):

    # Step 1: Compute spatial distance matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed', projection_params=projection_params)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    # Step 2: Apply filtering to each FC matrix
    fcs_filtered = []
    for fc in fcs:
        filtered = apply_laplacian_filter(matrix=fc, distance_matrix=distance_matrix, 
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
    sample = utils_feature_loading.read_fcs_mat(dataset='seed', identifier='sub1ex1', feature='pcc')['gamma']
    sample_average = np.mean(sample, axis=0)
    utils_visualization.draw_projection(sample_average)
    
    # gaussian smooth
    cm_filtered = fcs_spatial_filter(sample, projection_params={"source": "auto", "type": "3d_spherical"},
                                     filter_params={'computation': 'gaussian', 'lateral_mode': 'bilateral',
                                                    'sigma': 0.1, 'lambda_reg': None, 'reinforce': False}, 
                                     visualize=True)
    
    # inverse gaussian; lambda = 0.1
    cm_filtered = fcs_spatial_filter(sample, projection_params={"source": "auto", "type": "3d_spherical"}, 
                                     filter_params={'computation': 'pseudoinverse', 'lateral_mode': 'bilateral',
                                                    'sigma': 0.1, 'lambda_reg':0.1, 'reinforce': False}, 
                                     visualize=True)
    
    # inverse gaussian; lambda = 0.01
    cm_filtered = fcs_spatial_filter(sample, projection_params={"source": "auto", "type": "3d_spherical"}, 
                                     filter_params={'computation': 'pseudoinverse', 'lateral_mode': 'bilateral',
                                                    'sigma': 0.1, 'lambda_reg': 0.01, 'reinforce': False}, 
                                     visualize=True)
    
    # laplacian graph filtering
    cm_filtered = fcs_laplacian_geaph_filtering(sample, projection_params={"source": "auto", "type": "3d_spherical"},
                               filtering_params={'alpha': 0.1, 'lateral_mode': 'bilateral', 'normalized': False, 'reinforce': False}, 
                               visualize=True)
    
    # %% Channel Feature
    # de_sample = utils_feature_loading.read_cfs('seed', 'sub1ex1', 'de_LDS')
    # gamma = de_sample['gamma']
    
    # de_gamma_average = np.mean(gamma, axis=0)
    # utils_visualization.draw_heatmap_1d(de_gamma_average)
    
    # de_gamma_smoothed_sample = cfs_gaussian_filtering(gamma)
    # de_gamma_smoothed_average = np.mean(de_gamma_smoothed_sample, axis=0)
    # utils_visualization.draw_heatmap_1d(de_gamma_smoothed_average)