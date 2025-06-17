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
def spatial_gaussian_smoothing_on_vector(A, distance_matrix, sigma):
    dists = distance_matrix
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))
    weights /= weights.sum(axis=1, keepdims=True)
    A_smooth = weights @ A
    return A_smooth

def spatial_gaussian_smoothing_on_fc_matrix(A, distance_matrix, sigma=None, visualize=False):
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
    A_smooth = weights @ A @ weights.T

    # Step 5 (optional): Enforce symmetry
    # A_smooth = 0.5 * (A_smooth + A_smooth.T)
    
    if visualize:
        try:
            utils_visualization.draw_projection(A_smooth, 'After Spatial Gaussian Smoothing')
        except ModuleNotFoundError: 
            print("utils_visualization not found")
    
    return A_smooth

def spatial_gaussian_smoothing_on_vector_(A, coordinates, sigma):
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T
    dists = cdist(coords, coords)
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))
    weights /= weights.sum(axis=1, keepdims=True)
    A_smooth = weights @ A
    return A_smooth

def spatial_gaussian_smoothing_on_fc_matrix_(A, coordinates, sigma, visualize=False):
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
    
    # Step 1: Stack coordinate vectors to (N, 3)
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T  # shape (N, 3)

    # Step 2: Compute Euclidean distance matrix between channels
    dists = cdist(coords, coords)  # shape (N, N)

    # Step 3: Compute spatial Gaussian weights
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))  # shape (N, N)
    weights /= weights.sum(axis=1, keepdims=True)       # normalize per row

    # Step 4: Apply spatial smoothing to both rows and columns
    A_smooth = weights @ A @ weights.T

    # Step 5 (optional): Enforce symmetry
    # A_smooth = 0.5 * (A_smooth + A_smooth.T)
    
    if visualize:
        try:
            utils_visualization.draw_projection(A_smooth, 'After Spatial Gaussian Smoothing')
        except ModuleNotFoundError: 
            print("utils_visualization not found")
    
    return A_smooth

def fcs_spatial_gaussian_smoothing(fcs, projection_params={"source": "auto", "type": "3d"}, sigma=0.05):
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

def cfs_spatial_gaussian_smoothing(cfs, projection_params={"source": "auto", "type": "3d"}, sigma=0.05):
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

if __name__ == '__main__':  
    # %% Distance Matrix
    _, distance_matrix_2d_manual = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                        projection_params={"source": "manual", "type": "2d"})
    distance_matrix_2d_manual = feature_engineering.normalize_matrix(distance_matrix_2d_manual)
    utils_visualization.draw_projection(distance_matrix_2d_manual)
    
    _, distance_matrix_2d = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                        projection_params={"source": "auto", "type": "2d"})
    distance_matrix_2d = feature_engineering.normalize_matrix(distance_matrix_2d)
    utils_visualization.draw_projection(distance_matrix_2d)
    
    _, distance_matrix_3d = feature_engineering.compute_distance_matrix(dataset="seed", 
                                                                        projection_params={"source": "auto", "type": "3d"})
    distance_matrix_3d = feature_engineering.normalize_matrix(distance_matrix_3d)
    utils_visualization.draw_projection(distance_matrix_3d)

    _, distance_matrix_ste = feature_engineering.compute_distance_matrix('seed', 
                                                        projection_params={"source": "auto", "type": "stereo"}, visualize=True)
    distance_matrix_ste = feature_engineering.normalize_matrix(distance_matrix_ste)
    utils_visualization.draw_projection(distance_matrix_ste)
    
    # %% Connectivity Matrix
    cm_pcc_sample = utils_feature_loading.read_fcs_mat(dataset='seed', identifier='sub1ex1', feature='pcc')
    gamma = cm_pcc_sample['gamma']

    cm_gamma_smoothed = fcs_spatial_gaussian_smoothing(gamma, {"source": "manual", "type": "2d"}, 0.05)
    cm_gamma_smoothed_average = np.mean(cm_gamma_smoothed, axis=0)
    utils_visualization.draw_projection(cm_gamma_smoothed_average)
    
    # %% Channel Feature
    de_sample = utils_feature_loading.read_cfs('seed', 'sub1ex1', 'de_LDS')
    gamma = de_sample['gamma']
    
    de_gamma_average = np.mean(gamma, axis=0)
    utils_visualization.draw_heatmap_1d(de_gamma_average)
    
    de_gamma_smoothed_sample = cfs_spatial_gaussian_smoothing(gamma)
    de_gamma_smoothed_average = np.mean(de_gamma_smoothed_sample, axis=0)
    utils_visualization.draw_heatmap_1d(de_gamma_smoothed_average)