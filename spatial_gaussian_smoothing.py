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

# generalized surface laplacian filtering
def apply_generalized_surface_laplacian_filtering(matrix, distance_matrix,
                                                  filtering_params=None,
                                                  visualize=False,
                                                  upper_only=True):
    """
    向量化加速版（等价于原实现）。时间复杂度依然 ~O(N^3)，
    但去掉了 Python 级三层循环，通常能快一个数量级以上。
    """
    if filtering_params is None:
        filtering_params = {'sigma': 0.1, 'reinforce': False}

    sigma = filtering_params.get('sigma', 0.1)
    reinforce = filtering_params.get('reinforce', False)

    M = np.asarray(matrix, dtype=float)
    D = np.asarray(distance_matrix, dtype=float)
    N = M.shape[0]

    # 避免0距离（与你原实现一致）
    D = np.where(D == 0, 1e-6, D)

    F = np.zeros_like(M)

    inv2sig2 = 1.0 / (2.0 * sigma * sigma)

    # 可选：只计算上三角，最后镜像
    jslice = lambda i: slice(i+1, N) if upper_only else slice(0, N)

    for i in range(N):
        # ---- 第二部分：sum_k w2_i(k) * M[k, j]（与 j 无关，先整体算）----
        # w2_i(k) = exp(-(D[i,k]^2)/(2σ^2))
        w2_i = np.exp(- (D[i, :] ** 2) * inv2sig2)  # 形状 (N,)

        # 形成一整行的第二部分贡献：对于所有 j，lap2[j] = sum_k w2_i(k) * M[k, j]
        # 即 lap2 = w2_i^T @ M
        lap2 = w2_i @ M  # 形状 (N,)

        # ---- 第一部分：sum_k w1_ij(k) * M[i, k]（对每个 j 的权随 j 变化）----
        # 为固定的 i，构造 W1(j,k) = exp(-((D[i,k] + D[j,k])^2)/(2σ^2))
        # 这里用广播生成 N×N 的权重矩阵（行对应 j，列对应 k）
        Di = D[i, :][None, :]       # 1×N
        Dj = D[:, :][:, :]          # N×N（只是为了表达清晰，等价于 D）
        # 利用广播：Dj[:, k] + Di[:, k]
        W1 = np.exp(- ((Di + Dj) ** 2) * inv2sig2)  # 形状 N×N

        # 对每个 j：lap1[j] = sum_k W1[j,k] * M[i,k]，即矩阵–向量乘
        Mi_row = M[i, :]  # 形状 (N,)
        lap1 = W1 @ Mi_row  # 形状 (N,)

        # 合并两部分并减去
        jidx = np.arange(N)
        F[i, jidx] = M[i, jidx] - (lap1 + lap2)

        # 跳过对角
        F[i, i] = 0.0

        # 如果只算上三角，则当前 i 只保留 j>i，其他位置先置0
        if upper_only:
            F[i, :i+1] = 0.0

    if upper_only:
        # 镜像成对称（你原矩阵标注为对称）
        F = F + F.T - np.diag(np.diag(F))

    if reinforce:
        F += M

    # 可选可视化（保留你的调用接口）
    if visualize:
        try:
            utils_visualization.draw_projection(matrix, 'Before FN-Laplacian Filtering')
            utils_visualization.draw_projection(F, 'After FN-Laplacian Filtering (NumPy)')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return F

# generalized surface laplacian filtering
def apply_generalized_surface_laplacian_filtering_(matrix, distance_matrix,
                                                   filtering_params={'computation': 'generalized_surface_laplacian_filtering',
                                                                     'sigma': 0.1,
                                                                     'reinforce': False},
                                                   visualize=False):
    """
    Applies a Laplacian-style residual filter to a functional connectivity (FC) matrix,
    extending the idea of EEG surface Laplacian to connectivity edges.
    
    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Input functional connectivity matrix (symmetric).
    distance_matrix : np.ndarray, shape (N, N)
        Pairwise distance matrix between channels.
    filtering_params : dict
        Filtering parameters:
            - 'sigma': float, spatial Gaussian kernel width for defining edge-edge weights.
            - 'reinforce': bool, if True, add original FN back to filtered result.
    visualize : bool
        If True, visualize before and after matrices.

    Returns
    -------
    filtered_matrix : np.ndarray
        Laplacian-filtered connectivity matrix.
    """

    if visualize:
        try:
            utils_visualization.draw_projection(matrix, 'Before FN-Laplacian Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    sigma = filtering_params.get('sigma', 0.1)

    # Avoid zero distances
    distance_matrix = np.where(distance_matrix == 0, 1e-6, distance_matrix)
    
    N = matrix.shape[0]
    filtered_matrix = np.zeros_like(matrix)

    # ---- Step 1: Define weights between edges (i,j) and (k,l) ----
    # Here, N(i,j) = edges that share a node with (i,j)
    for i in range(N):
        for j in range(N):
            if i == j: 
                continue
            neighbors = set([(i, k) for k in range(N) if k != i]) | set([(k, j) for k in range(N) if k != j])
            val = matrix[i, j]
            lap_term = 0.0
            for (k, l) in neighbors:
                if k == l:
                    continue
                # Gaussian weight based on node distances
                dist = distance_matrix[i, k] + distance_matrix[j, l]
                w = np.exp(-(dist**2) / (2 * sigma**2))
                lap_term += w * matrix[k, l]
            filtered_matrix[i, j] = val - lap_term

    # ---- Step 2: Reinforce original if requested ----
    if filtering_params.get('reinforce', False):
        filtered_matrix += matrix

    if visualize:
        try:
            utils_visualization.draw_projection(filtered_matrix, 'After FN-Laplacian Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return filtered_matrix

# laplacian graph filtering
def apply_graph_laplacian_filtering(matrix, distance_matrix,
                                 filtering_params={'computation': 'graph_laplacian_filtering',
                                                   'alpha': 1,
                                                   'sigma': None,  # 新增
                                                   'lateral_mode': 'bilateral',
                                                   'normalized': False,
                                                   'reinforce': False},
                                 visualize=False):
    """
    ...
    alpha : 为简化模型设置为1
    sigma : float or None
        高斯核的尺度参数。如果为 None，则默认取非零距离的均值。
    """

    alpha = filtering_params.get('alpha', 1)
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
def apply_truncated_graph_spectral_filtering(matrix, distance_matrix,
                                             filtering_params={'computation': 'truncated_graph_spectral_filtering',
                                                               'cutoff': 0.1,
                                                               'mode': 'lowpass',  # 可选 'lowpass' 或 'highpass'
                                                               'normalized': False,
                                                               'reinforce': False},
                                             visualize=False):
    """
    Graph Spectral Filtering (low-pass / high-pass) for FC matrices.

    Parameters
    ----------
    cutoff : float in (0,1] or int >= 1
        - float -> proportion of lowest-frequency modes (rate-based)
        - int   -> number of lowest-frequency modes (rank-based)
        Backward-compat:
          * accepts `cutoff_rate` (float) or `cutoff_rank` (int).
          * if both given, `cutoff` > `cutoff_rate` > `cutoff_rank` in precedence.
    mode : {'lowpass', 'highpass'}
        'lowpass': keep the lowest-frequency k modes.
        'highpass': remove the lowest-frequency k modes.
    normalized : bool
        Use normalized Laplacian if True.
    reinforce : bool
        Add residual (original matrix) after filtering.

    Notes
    -----
    - Using a *rate* is recommended when N varies across datasets/montages.
    - k is clamped to [1, N-1] to avoid degenerate projectors.
    """
    import numpy as np

    if filtering_params is None:
        filtering_params = {}

    mode = filtering_params.get('mode', 'lowpass').lower()
    normalized = filtering_params.get('normalized', False)
    reinforce = filtering_params.get('reinforce', False)

    # ----- cutoff resolution (unified) -----
    cutoff = filtering_params.get('cutoff', None)
    if cutoff is None:
        # fallback to explicit legacy keys
        if 'cutoff_rate' in filtering_params:
            cutoff = float(filtering_params['cutoff_rate'])
        elif 'cutoff_rank' in filtering_params:
            cutoff = int(filtering_params['cutoff_rank'])
        else:
            cutoff = 0.1  # sensible default: keep/remove ~10% of modes

    N = matrix.shape[0]
    if isinstance(cutoff, float):
        if not (0 < cutoff <= 1):
            raise ValueError("cutoff as float must be in (0,1].")
        k = max(1, min(N-1, int(round(cutoff * N))))
    elif isinstance(cutoff, int):
        k = max(1, min(N-1, int(cutoff)))
    else:
        raise TypeError("cutoff must be float in (0,1] or int >= 1.")

    if visualize:
        try:
            utils_visualization.draw_projection(matrix, 'Before Graph Spectral Filtering')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    # ----- Graph construction -----
    dm = np.array(distance_matrix, dtype=float)
    dm = np.where(dm == 0, 1e-6, dm)
    sigma = np.mean(dm[dm > 0])
    W = np.exp(-np.square(dm) / (2 * sigma ** 2))

    # Laplacian
    D = np.diag(W.sum(axis=1))
    if normalized:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W

    # ----- Eigen-decomposition -----
    evals, evecs = np.linalg.eigh(L)
    order = np.argsort(evals)
    evecs = evecs[:, order]
    # low-frequency basis
    U_low = evecs[:, :k]
    P_low = U_low @ U_low.T
    I = np.eye(N)

    # ----- Filtering -----
    if mode == 'lowpass':
        filtered = P_low @ matrix @ P_low.T
    elif mode == 'highpass':
        P_high = I - P_low
        filtered = P_high @ matrix @ P_high.T
    else:
        raise ValueError("mode must be 'lowpass' or 'highpass'.")

    if reinforce:
        filtered += matrix

    if visualize:
        try:
            utils_visualization.draw_projection(filtered, f'After Graph Spectral Filtering ({mode}, k={k})')
        except ModuleNotFoundError:
            print("Visualization module not found.")

    return filtered

# spectral graph filtering
def apply_exp_graph_spectral_filtering(matrix, distance_matrix,
                                   filtering_params={'computation': 'exp_graph_spectral_filtering',
                                                     't': 10,
                                                     'mode': 'lowpass',   # 可选 'lowpass' 或 'highpass'
                                                     'normalized': False,
                                                     'reinforce': False},
                                   visualize=False):
    """
    Applies Exponential Graph Spectral Filtering (Low-pass or High-pass) to a 
    functional connectivity matrix in the graph Laplacian spectrum.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Input functional connectivity matrix (e.g., PCC or PLV).
    distance_matrix : np.ndarray, shape (N, N)
        Pairwise spatial distances between EEG channels.
    t : float
        Controls the strength of filtering. Larger t makes the effect stronger.
        - In low-pass mode: suppresses high frequencies more.
        - In high-pass mode: removes more smooth components.
    mode : str
        'lowpass' (h(λ)=exp(-tλ)) or 'highpass' (h(λ)=1-exp(-tλ)).
    normalized : bool
        If True, use normalized Laplacian; otherwise unnormalized.
    reinforce : bool
        If True, adds original matrix back to filtered result (residual enhancement).
    visualize : bool
        If True, show pre- and post-filter visualization.

    Returns
    -------
    filtered_matrix : np.ndarray
        The spectrally filtered functional connectivity matrix.
    """
    t = filtering_params.get('t', 10)
    mode = filtering_params.get('mode', 'lowpass').lower()
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

    # Step 4: Construct exponential filter
    if mode == 'lowpass':
        h_lambda = np.exp(-t * eigenvalues)      # Low-pass: keep smooth components
    elif mode == 'highpass':
        h_lambda = 1 - np.exp(-t * eigenvalues)  # High-pass: remove smooth components
    else:
        raise ValueError("Invalid mode. Use 'lowpass' or 'highpass'.")

    H = eigenvectors @ np.diag(h_lambda) @ eigenvectors.T

    # Step 5: Filter matrix: apply H on both sides
    filtered_matrix = H @ matrix @ H.T

    # Step 6: Optional residual reinforcement
    if reinforce:
        filtered_matrix += matrix

    if visualize:
        try:
            utils_visualization.draw_projection(
                filtered_matrix, f'After Exp Graph Spectral Filtering ({mode})'
            )
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

# %% aplly filters on fcs
def fcs_filtering_common(fcs,
                         projection_params={"source": "auto", "type": "3d_spherical"},
                         filtering_params={}, 
                         apply_filter='diffusion_inverse',
                         visualize=False):
    # valid filters
    filters_valid={'gaussian_filtering', 
                   'diffusion_inverse', 'graph_laplacian_filtering', 
                   'generalized_surface_laplacian_filtering',
                   'truncated_graph_spectral_filtering', 'exp_graph_spectral_filtering',
                   'graph_tikhonov_inverse'}
    
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
        elif apply_filter=='generalized_surface_laplacian_filtering':
            fcs_filtered = apply_generalized_surface_laplacian_filtering(matrix=fcs, distance_matrix=distance_matrix, 
                                                                         filtering_params=filtering_params,
                                                                         visualize=False)
        elif apply_filter=='graph_laplacian_filtering':
            fcs_filtered=apply_graph_laplacian_filtering(matrix=fcs, distance_matrix=distance_matrix, 
                                                      filtering_params=filtering_params,
                                                      visualize=False)
        elif apply_filter=='truncated_graph_spectral_filtering':
            fcs_filtered=apply_truncated_graph_spectral_filtering(matrix=fcs, distance_matrix=distance_matrix, 
                                                                  filtering_params=filtering_params,
                                                                  visualize=False)
        elif apply_filter=='exp_graph_spectral_filtering':
            fcs_filtered=apply_exp_graph_spectral_filtering(matrix=fcs, distance_matrix=distance_matrix, 
                                                            filtering_params=filtering_params,
                                                            visualize=False)
        elif apply_filter=='graph_tikhonov_inverse':
            fcs_filtered=apply_graph_tikhonov_inverse(matrix=fcs, distance_matrix=distance_matrix, 
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
        
        elif apply_filter=='generalized_surface_laplacian_filtering':
            for fc in fcs:
                filtered = apply_generalized_surface_laplacian_filtering(matrix=fc, distance_matrix=distance_matrix, 
                                                                         filtering_params=filtering_params,
                                                                         visualize=False)
                fcs_filtered.append(filtered)
        
        elif apply_filter=='truncated_graph_spectral_filtering':
            for fc in fcs:
                filtered = apply_truncated_graph_spectral_filtering(matrix=fc, distance_matrix=distance_matrix, 
                                                                    filtering_params=filtering_params,
                                                                    visualize=False)
                fcs_filtered.append(filtered)
        
        elif apply_filter=='exp_graph_spectral_filtering':
            for fc in fcs:
                filtered = apply_exp_graph_spectral_filtering(matrix=fc, distance_matrix=distance_matrix, 
                                                              filtering_params=filtering_params,
                                                              visualize=False)
                fcs_filtered.append(filtered)
        
        elif apply_filter=='graph_tikhonov_inverse':
            for fc in fcs:
                filtered = apply_graph_tikhonov_inverse(matrix=fc, distance_matrix=distance_matrix, 
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
    
    # generalized_surface_laplacian_filtering; sigma = 0.1
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'generalized_surface_laplacian_filtering',
                                                         'sigma': 0.1,
                                                         'lateral_mode': 'bilateral', 'normalized': False, 'reinforce': False}, 
                                       apply_filter='generalized_surface_laplacian_filtering',
                                       visualize=True)
    
    # graph_laplacian_filtering; alpha = 0.1
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'graph_laplacian_filtering',
                                                         'alpha': 0.1, 'sigma': None,
                                                         'lateral_mode': 'bilateral', 'normalized': False, 'reinforce': False}, 
                                       apply_filter='graph_laplacian_filtering',
                                       visualize=True)
    
    # graph_spectral_filtering; cutoff = 0.1
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'truncated_graph_spectral_filtering',
                                                         'cutoff': 0.1, 'mode': 'highpass',
                                                         'normalized': False, 'reinforce': False}, 
                                       apply_filter='truncated_graph_spectral_filtering',
                                       visualize=True)
    
    #  exp_graph_spectral_filtering; t = 10
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                         't': 10, 'mode': 'lowpass',
                                                         'normalized': False, 'reinforce': False}, 
                                       apply_filter='exp_graph_spectral_filtering',
                                       visualize=True)
    
    # graph_tikhonov_inverse; alpha = 0.1, lambda = 1e-2
    cm_filtered = fcs_filtering_common(sample_averaged,
                                       projection_params={"source": "auto", "type": "3d_spherical"},
                                       filtering_params={'computation': 'graph_tikhonov_inverse',
                                                         'alpha': 0.1, 'lambda': 1e-2,
                                                         'normalized': False, 'reinforce': False}, 
                                       apply_filter='graph_tikhonov_inverse',
                                       visualize=True)
    
    # %% Channel Feature
    # de_sample = utils_feature_loading.read_cfs('seed', 'sub1ex1', 'de_LDS')
    # gamma = de_sample['gamma']
    
    # de_gamma_average = np.mean(gamma, axis=0)
    # utils_visualization.draw_heatmap_1d(de_gamma_average)
    
    # de_gamma_smoothed_sample = cfs_gaussian_filtering(gamma)
    # de_gamma_smoothed_average = np.mean(de_gamma_smoothed_sample, axis=0)
    # utils_visualization.draw_heatmap_1d(de_gamma_smoothed_average)