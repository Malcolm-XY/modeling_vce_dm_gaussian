# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:21:23 2025

@author: usouu
"""
import os
import numpy as np
import pandas as pd

import torch

import cnn_validation
from models import models
from utils import utils_feature_loading

# %% read parameters/save
def read_params(model='exponential', model_fm='basic', model_rcm='differ', folder='fitting_results(15_15_joint_band_from_mat)'):
    identifier = f'{model_fm.lower()}_fm_{model_rcm.lower()}_rcm'
    
    path_current = os.getcwd()
    path_fitting_results = os.path.join(path_current, 'fitting_results', folder)
    file_path = os.path.join(path_fitting_results, f'fitting_results({identifier}).xlsx')
    
    df = pd.read_excel(file_path).set_index('method')
    df_dict = df.to_dict(orient='index')
    
    model = model.upper()
    params = df_dict[model]
    
    return params

def save_to_xlsx_sheet(df, folder_name, file_name, sheet_name):
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    # Append or create the Excel file
    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    
def save_to_xlsx_fitting(results, subject_range, experiment_range, folder_name, file_name, sheet_name):
    # calculate average
    result_keys = results[0].keys()
    avg_results = {key: np.mean([res[key] for res in results]) for key in result_keys}
    
    # save to xlsx
    # 准备结果数据
    df_results = pd.DataFrame(results)
    df_results.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in subject_range for j in experiment_range])
    df_results.loc["Average"] = ["Average"] + list(avg_results.values())
    
    # 构造保存路径
    path_save = os.path.join(os.getcwd(), folder_name, file_name)
    
    # 判断文件是否存在
    if os.path.exists(path_save):
        # 追加模式，保留已有 sheet，添加新 sheet
        with pd.ExcelWriter(path_save, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # 新建文件
        with pd.ExcelWriter(path_save, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)

# %% Executor
def cnn_subnetworks_evaluation_circle_original_cm(selection_rate=1, feature_cm='pcc', 
                                                 subject_range=range(11,16), experiment_range=range(1,4), 
                                                 save=False):
    functional_node_strength = {
        'alpha': [],
        'beta': [],
        'gamma': []
    }
    
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']
            
            # Compute node strength
            strength_alpha = np.sum(np.abs(alpha), axis=1)
            strength_beta = np.sum(np.abs(beta), axis=1)
            strength_gamma = np.sum(np.abs(gamma), axis=1)
            
            # Save for further analysis
            functional_node_strength['alpha'].append(strength_alpha)
            functional_node_strength['beta'].append(strength_beta)
            functional_node_strength['gamma'].append(strength_gamma)
    
    # channel weights computed from the rebuilded connectivity matrix that constructed by vce modeling
    channel_weights = {'gamma': np.mean(np.mean(functional_node_strength['gamma'], axis=0), axis=0),
                       'beta': np.mean(np.mean(functional_node_strength['beta'], axis=0), axis=0),
                       'alpha': np.mean(np.mean(functional_node_strength['alpha'], axis=0), axis=0)
                       }
    
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    
    # for traning and testing in CNN
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
   
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # Selected CM           
            alpha_selected = alpha[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_selected = beta[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_selected = gamma[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_selected = np.stack((alpha_selected, beta_selected, gamma_selected), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_selected, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save
    if save:        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_origin.xlsx'
        sheet_name = f'sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

import spatial_gaussian_smoothing
def cnn_subnetworks_evaluation_circle_rebuilt_cm(projection_params={"source": "auto", "type": "3d_euclidean"},
                                                 filter_params={'computation': 'pseudoinverse', 'lateral_mode': 'bilateral',
                                                                'sigma': 0.1, 'lambda_reg': 0.01, 'reinforce': False}, 
                                                 selection_rate=1, feature_cm='pcc', 
                                                 subject_range=range(11,16), experiment_range=range(1,4), 
                                                 save=False):
    functional_node_strength = {
        'alpha': [],
        'beta': [],
        'gamma': []
    }
    
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # RCM           
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_spatial_filter(alpha, projection_params, filter_params)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_spatial_filter(beta, projection_params, filter_params)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_spatial_filter(gamma, projection_params, filter_params)
            
            # Compute node strength
            strength_alpha = np.sum(np.abs(alpha_rebuilded), axis=1)
            strength_beta = np.sum(np.abs(beta_rebuilded), axis=1)
            strength_gamma = np.sum(np.abs(gamma_rebuilded), axis=1)
            
            # Save for further analysis
            functional_node_strength['alpha'].append(strength_alpha)
            functional_node_strength['beta'].append(strength_beta)
            functional_node_strength['gamma'].append(strength_gamma)
    
    # channel weights computed from the rebuilded connectivity matrix that constructed by vce modeling
    channel_weights = {'gamma': np.mean(np.mean(functional_node_strength['gamma'], axis=0), axis=0),
                       'beta': np.mean(np.mean(functional_node_strength['beta'], axis=0), axis=0),
                       'alpha': np.mean(np.mean(functional_node_strength['alpha'], axis=0), axis=0)
                       }
    
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    
    # for traning and testing in CNN
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # RCM
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_spatial_filter(alpha, projection_params, filter_params)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_spatial_filter(beta, projection_params, filter_params)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_spatial_filter(gamma, projection_params, filter_params)
            
            alpha_rebuilded = alpha_rebuilded[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_rebuilded = beta_rebuilded[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_rebuilded = gamma_rebuilded[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_rebuilded = np.stack((alpha_rebuilded, beta_rebuilded, gamma_rebuilded), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuilded, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save
    if save:
        prj_type = projection_params.get('type')
        computation = filter_params.get('computation')
        sigma = filter_params.get('sigma')
        lambda_reg = filter_params.get('lambda_reg')
        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{prj_type}_{computation}_sigma_{sigma}_lamda_{lambda_reg}.xlsx'
        sheet_name = f'{computation}_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

def cnn_subnetworks_evaluation_circle_laplacian_cm(projection_params={"source": "auto", "type": "3d_euclidean"},
                                                   filtering_params={'computation': 'laplacian', 'lateral_mode': 'bilateral', 
                                                                     'alpha': 0.1, 'normalized': False, 'reinforce': False},
                                                 selection_rate=1, feature_cm='pcc', 
                                                 subject_range=range(11,16), experiment_range=range(1,4), 
                                                 save=False):
    functional_node_strength = {
        'alpha': [],
        'beta': [],
        'gamma': []
    }
    
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # RCM           
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_geaph_filtering(alpha, projection_params, filtering_params)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_geaph_filtering(beta, projection_params, filtering_params)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_geaph_filtering(gamma, projection_params, filtering_params)
            
            # Compute node strength
            strength_alpha = np.sum(np.abs(alpha_rebuilded), axis=1)
            strength_beta = np.sum(np.abs(beta_rebuilded), axis=1)
            strength_gamma = np.sum(np.abs(gamma_rebuilded), axis=1)
            
            # Save for further analysis
            functional_node_strength['alpha'].append(strength_alpha)
            functional_node_strength['beta'].append(strength_beta)
            functional_node_strength['gamma'].append(strength_gamma)
    
    # channel weights computed from the rebuilded connectivity matrix that constructed by vce modeling
    channel_weights = {'gamma': np.mean(np.mean(functional_node_strength['gamma'], axis=0), axis=0),
                       'beta': np.mean(np.mean(functional_node_strength['beta'], axis=0), axis=0),
                       'alpha': np.mean(np.mean(functional_node_strength['alpha'], axis=0), axis=0)
                       }
    
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    
    # for traning and testing in CNN
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # RCM
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_geaph_filtering(alpha, projection_params, filtering_params)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_geaph_filtering(beta, projection_params, filtering_params)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_geaph_filtering(gamma, projection_params, filtering_params)
            
            alpha_rebuilded = alpha_rebuilded[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_rebuilded = beta_rebuilded[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_rebuilded = gamma_rebuilded[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_rebuilded = np.stack((alpha_rebuilded, beta_rebuilded, gamma_rebuilded), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuilded, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save
    if save:
        prj_type = projection_params.get('type')
        computation = filtering_params.get('computation')
        alpha = filtering_params.get('alpha')
        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{prj_type}_{computation}_alpha_{alpha}.xlsx'
        sheet_name = f'{computation}_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

def cnn_subnetworks_evaluation_circle_rebuilt_cm_(projection_params={"source": "auto", "type": "3d_euclidean"},
                                                 filtering_type={'residual_type': 'origin', 'lateral_mode': 'bilateral'},
                                                 filtering_params={'sigma': 0.125, 'gamma': 0.25, 'lambda_reg': 0.25, 'reinforce': False},
                                                 selection_rate=1, feature_cm='pcc', 
                                                 subject_range=range(11,16), experiment_range=range(1,4), 
                                                 save=False):
    functional_node_strength = {
        'alpha': [],
        'beta': [],
        'gamma': []
    }
    
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # RCM
            residual_type = filtering_type.get('residual_type')
            lateral_mode = filtering_type.get('lateral_mode', 'bilateral')
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_residual_filtering(alpha, projection_params, 
                                                                                residual_type, lateral_mode, filtering_params)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_residual_filtering(beta, projection_params, 
                                                                                residual_type, lateral_mode, filtering_params)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_residual_filtering(gamma, projection_params, 
                                                                                residual_type, lateral_mode, filtering_params)
            
            # Compute node strength
            strength_alpha = np.sum(np.abs(alpha_rebuilded), axis=1)
            strength_beta = np.sum(np.abs(beta_rebuilded), axis=1)
            strength_gamma = np.sum(np.abs(gamma_rebuilded), axis=1)
            
            # Save for further analysis
            functional_node_strength['alpha'].append(strength_alpha)
            functional_node_strength['beta'].append(strength_beta)
            functional_node_strength['gamma'].append(strength_gamma)
    
    # channel weights computed from the rebuilded connectivity matrix that constructed by vce modeling
    channel_weights = {'gamma': np.mean(np.mean(functional_node_strength['gamma'], axis=0), axis=0),
                       'beta': np.mean(np.mean(functional_node_strength['beta'], axis=0), axis=0),
                       'alpha': np.mean(np.mean(functional_node_strength['alpha'], axis=0), axis=0)
                       }
    
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    
    # for traning and testing in CNN
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # RCM
            residual_type = filtering_type.get('residual_type')
            lateral_mode = filtering_type.get('lateral_mode', 'bilateral')
            
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_residual_filtering(alpha, projection_params, 
                                                                                residual_type, lateral_mode, filtering_params)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_residual_filtering(beta, projection_params, 
                                                                                residual_type, lateral_mode, filtering_params)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_residual_filtering(gamma, projection_params, 
                                                                                residual_type, lateral_mode, filtering_params)
            
            alpha_rebuilded = alpha_rebuilded[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_rebuilded = beta_rebuilded[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_rebuilded = gamma_rebuilded[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_rebuilded = np.stack((alpha_rebuilded, beta_rebuilded, gamma_rebuilded), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuilded, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save
    if save:
        prj_type = projection_params.get('type')
        residual_type = filtering_type.get('residual_type')
        sigma = filtering_params.get('sigma')
        lambda_reg = filtering_params.get('lambda_reg')
        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{prj_type}_{residual_type}_sigma_{sigma}_lamda_{lambda_reg}.xlsx'
        sheet_name = f'{residual_type}_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

# %% Execute
def parameters_optimization():
    # kernel_list = list(['origin', 'origin_gaussian', 'inverse', 'residual_mean', 'pseudoinverse'])
    # kernel_list = list(['origin', 'origin_gaussian', 'pseudoinverse'])
    # selection_rate_list = [1, 0.5, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07]
    
    selection_rate_list = [0.3, 0.2, 0.1]
    
    sigma_candidates = [0.3, 0.2, 0.15, 0.1, 0.05, 0.01]
    lambda_candidates = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    
    for selection_rate in selection_rate_list:      
        # cnn_subnetworks_evaluation_circle_rebuilt_cm(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                              filtering_type={'residual_type': 'origin_gaussian'},
        #                                              filtering_params={'sigma': 0.1, 'lambda_reg': 0.1, 'reinforce': False},
        #                                              selection_rate=selection_rate, feature_cm='pcc', save=True)
        
        # cnn_subnetworks_evaluation_circle_rebuilt_cm(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                              filtering_type={'residual_type': 'pseudoinverse'},
        #                                              filtering_params={'sigma': 0.1, 'lambda_reg': 0.05, 'reinforce': False},
        #                                              selection_rate=selection_rate, feature_cm='pcc', save=True)
        
        for sigma in sigma_candidates:
            for lam in lambda_candidates:
        
                cnn_subnetworks_evaluation_circle_rebuilt_cm(projection_params={"source": "auto", "type": "3d_spherical"},
                                                             filtering_type={'residual_type': 'pseudoinverse'},
                                                             filtering_params={'sigma': sigma, 'lambda_reg': lam, 'reinforce': False},
                                                             selection_rate=selection_rate, feature_cm='pcc', save=True)

if __name__ == '__main__':
    selection_rate_list = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    
    # optimized parameters
    sigma, lamda = 0.1, 0.01
    
    for selection_rate in selection_rate_list:
        # cnn_subnetworks_evaluation_circle_original_cm(selection_rate=selection_rate, feature_cm='pli', save=True)
        
        # cnn_subnetworks_evaluation_circle_rebuilt_cm(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                              filter_params={'computation': 'pseudoinverse', 'lateral_mode': 'bilateral',
        #                                                             'sigma': 0.1, 'lambda_reg': 0.01, 'reinforce': False},
        #                                              selection_rate=selection_rate, feature_cm='plv', save=True)
        
        cnn_subnetworks_evaluation_circle_laplacian_cm(projection_params={"source": "auto", "type": "3d_spherical"},
                                                       filtering_params={'alpha': 0.1, 'normalized': False, 'reinforce': False},
                                                       selection_rate=selection_rate, feature_cm='pcc', save=True)

    # %% End
    from cnn_val_circle import end_program_actions
    end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)