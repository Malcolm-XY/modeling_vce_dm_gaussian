# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 13:22:03 2025

@author: admin
"""

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
                                                 subject_range=range(6,16), experiment_range=range(1,4), 
                                                 subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
                                                 save=False):
    if subnetworks_extract == 'read':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=1)
        strength_beta = np.sum(np.abs(beta_global_averaged), axis=1)
        strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=1)
        
        channel_weights = {'alpha': strength_alpha, 
                           'beta': strength_beta,
                           'gamma': strength_gamma,
                           }

    elif subnetworks_extract == 'calculation':
        functional_node_strength = {'alpha': [], 'beta': [], 'gamma': []}
        
        for sub in subnetworks_exrtact_basis:
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
                                                 subject_range=range(6,16), experiment_range=range(1,4), 
                                                 subnetworks_extract='read', subnetworks_exrtact_basis=range(1, 6),
                                                 save=False):
    
    if subnetworks_extract == 'read':
        # ------test
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
    
        alpha_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_spatial_filter(alpha_global_averaged,
                                                                                        projection_params, filter_params)
        beta_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_spatial_filter(beta_global_averaged, 
                                                                                       projection_params, filter_params)
        gamma_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_spatial_filter(gamma_global_averaged,
                                                                                        projection_params, filter_params)
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
        strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
        strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)
        
        channel_weights = {'gamma': strength_gamma,
                           'beta': strength_beta,
                           'alpha': strength_alpha
                           }
        # ------end
    elif subnetworks_extract == 'calculation':
        # ------test
        functional_node_strength = {'alpha': [], 'beta': [], 'gamma': []}
        for sub in subnetworks_exrtact_basis:
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
        # ------end
        
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
                                                 subject_range=range(6,16), experiment_range=range(1,4), 
                                                 subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
                                                 save=False):
    if subnetworks_extract == 'read':    
        # ------test
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
    
        alpha_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_filtering(alpha_global_averaged,
                                                                                        projection_params, filtering_params)
        beta_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_filtering(beta_global_averaged, 
                                                                                       projection_params, filtering_params)
        gamma_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_filtering(gamma_global_averaged,
                                                                                        projection_params, filtering_params)
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
        strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
        strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)
        
        channel_weights = {'gamma': strength_gamma,
                           'beta': strength_beta,
                           'alpha': strength_alpha
                           }
        # ------end
    elif subnetworks_extract == 'calculation':
        # ------test
        functional_node_strength = {'alpha': [], 'beta': [], 'gamma': []}
        for sub in subnetworks_exrtact_basis:
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
                alpha_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_filtering(alpha, projection_params, filtering_params)
                beta_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_filtering(beta, projection_params, filtering_params)
                gamma_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_filtering(gamma, projection_params, filtering_params)
                
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
        # ------end
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
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_filtering(alpha, projection_params, filtering_params)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_filtering(beta, projection_params, filtering_params)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_filtering(gamma, projection_params, filtering_params)
            
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

def cnn_subnetworks_evaluation_circle_laplacian_denoising_cm(projection_params={"source": "auto", "type": "3d_euclidean"},
                                                             filtering_params={'computation': 'laplacian', 'cutoff_rank': 5,
                                                                               'normalized': False, 'reinforce': False},
                                                             selection_rate=1, feature_cm='pcc', 
                                                             subject_range=range(6,16), experiment_range=range(1,4), 
                                                             subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
                                                             save=False):
    if subnetworks_extract == 'read':    
        # ------test
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
    
        alpha_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_denoising(alpha_global_averaged,
                                                                                        projection_params, filtering_params)
        beta_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_denoising(beta_global_averaged, 
                                                                                       projection_params, filtering_params)
        gamma_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_denoising(gamma_global_averaged,
                                                                                        projection_params, filtering_params)
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
        strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
        strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)
        
        channel_weights = {'gamma': strength_gamma,
                           'beta': strength_beta,
                           'alpha': strength_alpha
                           }
        # ------end
    elif subnetworks_extract == 'calculation':
        # ------test
        functional_node_strength = {'alpha': [], 'beta': [], 'gamma': []}
        for sub in subnetworks_exrtact_basis:
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
                alpha_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_denoising(alpha, projection_params, filtering_params)
                beta_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_denoising(beta, projection_params, filtering_params)
                gamma_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_denoising(gamma, projection_params, filtering_params)
                
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
        # ------end
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
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_denoising(alpha, projection_params, filtering_params)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_denoising(beta, projection_params, filtering_params)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_laplacian_graph_denoising(gamma, projection_params, filtering_params)
            
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
        cutoff_rank = filtering_params.get('cutoff_rank')
        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{prj_type}_{computation}_cutoff_rank_{cutoff_rank}.xlsx'
        sheet_name = f'{computation}_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

def cnn_subnetworks_evaluation_circle_graph_tikhonov_inverse(projection_params={"source": "auto", "type": "3d_euclidean"},
                                                             filtering_params={'alpha': 0.1, 'lambda': 1e-2, 
                                                                               'normalized': False, 'reinforce': False},
                                                             selection_rate=1, feature_cm='pcc', 
                                                             subject_range=range(6,16), experiment_range=range(1,4), 
                                                             subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
                                                             save=False):
    if subnetworks_extract == 'read':    
        # ------test
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
        alpha_global_averaged = fcs_global_averaged['alpha']
        print(alpha_global_averaged.shape)
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
    
        alpha_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_graph_tikhonov_inverse(alpha_global_averaged,
                                                                                        projection_params, filtering_params)
        beta_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_graph_tikhonov_inverse(beta_global_averaged, 
                                                                                       projection_params, filtering_params)
        gamma_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_graph_tikhonov_inverse(gamma_global_averaged,
                                                                                        projection_params, filtering_params)
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
        strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
        strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)
        
        channel_weights = {'gamma': strength_gamma,
                           'beta': strength_beta,
                           'alpha': strength_alpha
                           }
        # ------end
    elif subnetworks_extract == 'calculation':
        # ------test
        functional_node_strength = {'alpha': [], 'beta': [], 'gamma': []}
        for sub in subnetworks_exrtact_basis:
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
                print(alpha.shape)
                beta = features['beta']
                gamma = features['gamma']
    
                # RCM           
                alpha_rebuilded = spatial_gaussian_smoothing.fcs_graph_tikhonov_inverse(alpha, projection_params, filtering_params)
                beta_rebuilded = spatial_gaussian_smoothing.fcs_graph_tikhonov_inverse(beta, projection_params, filtering_params)
                gamma_rebuilded = spatial_gaussian_smoothing.fcs_graph_tikhonov_inverse(gamma, projection_params, filtering_params)
                
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
        # ------end
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
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_graph_tikhonov_inverse(alpha, projection_params, filtering_params)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_graph_tikhonov_inverse(beta, projection_params, filtering_params)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_graph_tikhonov_inverse(gamma, projection_params, filtering_params)
            
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
        alpha_ = filtering_params.get('alpha')
        lambda_ = filtering_params.get('lambda')
        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{prj_type}_{computation}_alpha_{alpha_}_lambda_{lambda_}.xlsx'
        sheet_name = f'{computation}_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

# %% cnn subnetworks evaluation circle common
def global_average_consistency_check(apply_filter='diffusion_inverse',
                                     selection_rate=1, feature_cm='pcc',
                                     subnetworks_exrtact_basis=range(1,6)):
    # projection
    projection_params={"source": "auto", "type": "3d_spherical"}
    
    # valid filters
    filters_valid={'diffusion_inverse', 'graph_laplacian', 'graph_laplacian_denoising', 'graph_tikhonov_inverse'}
    
    # default parameteres
    if apply_filter == 'diffusion_inverse':
        filtering_params={'computation': 'gaussian', 'lateral_mode': 'bilateral',
                       'sigma': None, 'lambda_reg': None, 'reinforce': False}
    elif apply_filter == 'graph_laplacian':
        filtering_params={'computation': 'laplacian_graph', 'lateral_mode': 'bilateral',
                          'alpha': 0.1, 'normalized': False, 'reinforce': False}
    elif apply_filter == 'graph_laplacian_denoising':
        filtering_params={'computation': 'laplacian_graph_denoising',
                          'cutoff_rank': 5, 'normalized': False, 'reinforce': False}
    elif apply_filter == 'graph_tikhonov_inverse':
        filtering_params={'computation': 'graph_tikhonov_inverse', 
                          'alpha': 0.1, 'lambda': 1e-2, 'normalized': False, 'reinforce': False}
    
    # ------read
    fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
    alpha_global_averaged = fcs_global_averaged['alpha']
    beta_global_averaged = fcs_global_averaged['beta']
    gamma_global_averaged = fcs_global_averaged['gamma']
    
    alpha_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(alpha_global_averaged,
                                                                                      projection_params, 
                                                                                      filtering_params,
                                                                                      apply_filter)
    beta_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(beta_global_averaged,
                                                                                     projection_params, 
                                                                                     filtering_params,
                                                                                     apply_filter)
    gamma_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(gamma_global_averaged,
                                                                                      projection_params, 
                                                                                      filtering_params,
                                                                                      apply_filter)
    
    strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=0)
    strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=0)
    strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=0)
    
    channel_weights_read = {'gamma': strength_gamma, 'beta': strength_beta, 'alpha': strength_alpha}
    # ------end
    
    # ------calculation
    alphas, betas, gammas = [], [], []
    experiment_range=range(1,4)
    for sub in subnetworks_exrtact_basis:
        print(f"Evaluating subject No.{sub}...")
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']
            
            alphas.append(alpha)
            betas.append(beta)
            gammas.append(gamma)
    
    alpha_global_averaged_cal = np.mean(np.mean(alphas, axis=0), axis=0)
    beta_global_averaged_cal = np.mean(np.mean(betas, axis=0), axis=0)
    gamma_global_averaged_cal = np.mean(np.mean(gammas, axis=0), axis=0)
    
    alpha_global_averaged_cal_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(alpha_global_averaged_cal,
                                                                                          projection_params, 
                                                                                          filtering_params,
                                                                                          apply_filter)
    beta_global_averaged_cal_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(beta_global_averaged_cal,
                                                                                         projection_params, 
                                                                                         filtering_params,
                                                                                         apply_filter)
    gamma_global_averaged_cal_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(gamma_global_averaged_cal,
                                                                                         projection_params, 
                                                                                         filtering_params,
                                                                                         apply_filter)
    
    strength_alpha_cal = np.sum(np.abs(alpha_global_averaged_cal_rebuilded), axis=1)
    strength_beta_cal = np.sum(np.abs(beta_global_averaged_cal_rebuilded), axis=1)
    strength_gamma_cal = np.sum(np.abs(gamma_global_averaged_cal_rebuilded), axis=1)
    
    channel_weights_calculation = {'gamma': strength_gamma_cal, 'beta': strength_beta_cal, 'alpha': strength_alpha_cal}
    # ------end
    
    k_read = {'gamma': int(len(channel_weights_read['gamma']) * selection_rate),
              'beta': int(len(channel_weights_read['beta']) * selection_rate),
              'alpha': int(len(channel_weights_read['alpha']) * selection_rate),
              }
    
    channel_selects_read = {'gamma': np.argsort(channel_weights_read['gamma'])[-k_read['gamma']:][::-1],
                            'beta': np.argsort(channel_weights_read['beta'])[-k_read['beta']:][::-1],
                            'alpha': np.argsort(channel_weights_read['alpha'])[-k_read['alpha']:][::-1]
                            }
    
    k_calculation = {'gamma': int(len(channel_weights_calculation['gamma']) * selection_rate),
                     'beta': int(len(channel_weights_calculation['beta']) * selection_rate),
                     'alpha': int(len(channel_weights_calculation['alpha']) * selection_rate),
                     }
    
    channel_selects_calculation = {'gamma': np.argsort(channel_weights_calculation['gamma'])[-k_calculation['gamma']:][::-1],
                                   'beta': np.argsort(channel_weights_calculation['beta'])[-k_calculation['beta']:][::-1],
                                   'alpha': np.argsort(channel_weights_calculation['alpha'])[-k_calculation['alpha']:][::-1]
                                   }
    
    if (channel_selects_read['gamma'] - channel_selects_calculation['gamma']).all()<1e-4:
        print(f'{apply_filter} gamma band consistency check ok!')
    else:
        print(f'{apply_filter} gamma band consistency check not ok')
    
    if (channel_selects_read['beta'] - channel_selects_calculation['beta']).all()<1e-4:
        print(f'{apply_filter} beta band consistency check ok!')
    else:
        print(f'{apply_filter} beta band consistency check not ok')
        
    if (channel_selects_read['alpha'] - channel_selects_calculation['alpha']).all()<1e-4:
        print(f'{apply_filter} alpha band consistency check ok!')
    else:
        print(f'{apply_filter} alpha band consistency check not ok')
    
    return channel_selects_read, channel_selects_calculation
        
def cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                             filtering_params={},
                                             selection_rate=1, feature_cm='pcc', 
                                             apply_filter='diffusion_inverse',
                                             subject_range=range(6,16), experiment_range=range(1,4),
                                             subnetworks_extract='read', 
                                             subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4), 
                                             save=False):
    # subnetworks selects;channel selects------start
    # valid filters
    filters_valid={'diffusion_inverse', 'graph_laplacian', 'graph_laplacian_denoising', 'graph_tikhonov_inverse'}
    
    # default parameteres
    # if apply_filter == 'diffusion_inverse':
    #     filtering_params={'computation': 'gaussian', 'lateral_mode': 'bilateral',
    #                    'sigma': None, 'lambda_reg': None, 'reinforce': False}
    # elif apply_filter == 'graph_laplacian':
    #     filtering_params={'computation': 'laplacian_graph', 'lateral_mode': 'bilateral',
    #                       'alpha': 0.1, 'normalized': False, 'reinforce': False}
    # elif apply_filter == 'graph_laplacian_denoising':
    #     filtering_params={'computation': 'laplacian_graph_denoising',
    #                       'cutoff_rank': 5, 'normalized': False, 'reinforce': False}
    # elif apply_filter == 'graph_tikhonov_inverse':
    #     filtering_params={'computation': 'graph_tikhonov_inverse', 
    #                       'alpha': 0.1, 'lambda': 1e-2, 'normalized': False, 'reinforce': False}
    
    if subnetworks_extract == 'read':
        # ------read
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, 'joint',
                                                                            subnets_exrtact_basis_sub)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
        
        alpha_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(alpha_global_averaged,
                                                                                          projection_params, 
                                                                                          filtering_params,
                                                                                          apply_filter)
        beta_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(beta_global_averaged,
                                                                                         projection_params, 
                                                                                         filtering_params,
                                                                                         apply_filter)
        gamma_global_averaged_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(gamma_global_averaged,
                                                                                          projection_params, 
                                                                                          filtering_params,
                                                                                          apply_filter)
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=0)
        strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=0)
        strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=0)
        
        channel_weights = {'gamma': strength_gamma, 'beta': strength_beta, 'alpha': strength_alpha}
        # ------end
    
    # ------calculation
    elif subnetworks_extract == 'calculation':
        alphas, betas, gammas = [], [], []
        for sub in subnets_exrtact_basis_sub:
            for ex in subnets_exrtact_basis_ex:
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
                
                alphas.append(alpha)
                betas.append(beta)
                gammas.append(gamma)
        
        alpha_global_averaged_cal = np.mean(np.mean(alphas, axis=0), axis=0)
        beta_global_averaged_cal = np.mean(np.mean(betas, axis=0), axis=0)
        gamma_global_averaged_cal = np.mean(np.mean(gammas, axis=0), axis=0)
        
        alpha_global_averaged_cal_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(alpha_global_averaged_cal,
                                                                                              projection_params, 
                                                                                              filtering_params,
                                                                                              apply_filter)
        beta_global_averaged_cal_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(beta_global_averaged_cal,
                                                                                             projection_params, 
                                                                                             filtering_params,
                                                                                             apply_filter)
        gamma_global_averaged_cal_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(gamma_global_averaged_cal,
                                                                                             projection_params, 
                                                                                             filtering_params,
                                                                                             apply_filter)
        
        strength_alpha_cal = np.sum(np.abs(alpha_global_averaged_cal_rebuilded), axis=1)
        strength_beta_cal = np.sum(np.abs(beta_global_averaged_cal_rebuilded), axis=1)
        strength_gamma_cal = np.sum(np.abs(gamma_global_averaged_cal_rebuilded), axis=1)
        
        channel_weights = {'gamma': strength_gamma_cal, 'beta': strength_beta_cal, 'alpha': strength_alpha_cal}
        # ------end
        
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    # subnetworks selects;channel selects------end
    
    # for traning and testing in CNN------start
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
            alpha_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(alpha, 
                                                                              projection_params, filtering_params,
                                                                              apply_filter)
            beta_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(beta, 
                                                                             projection_params, filtering_params,
                                                                             apply_filter)
            gamma_rebuilded = spatial_gaussian_smoothing.fcs_filtering_common(gamma, 
                                                                              projection_params, filtering_params,
                                                                              apply_filter)
            
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
        
        if apply_filter == 'diffusion_inverse':
            parameters_prompt = f'sigma_{filtering_params.get('sigma')}_lambda_reg_{filtering_params.get('lambda_reg')}'
        elif apply_filter == 'graph_laplacian':
            parameters_prompt = f'alpha_{filtering_params.get('alpha')}'
        elif apply_filter == 'graph_laplacian_denoising':
            parameters_prompt = f'cutoff_rank_{filtering_params.get('cutoff_rank')}'
        elif apply_filter == 'graph_tikhonov_inverse':
            parameters_prompt = f'alpha_{filtering_params.get('alpha')}_lambda_{filtering_params.get('lambda')}'
        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{prj_type}_{computation}_{parameters_prompt}.xlsx'
        sheet_name = f'{computation}_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

# %% Execute
def parameters_optimization():
    # selection_rate_list = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    
    selection_rate_list = [0.5, 0.3, 0.2, 0.1]
    
    for selection_rate in selection_rate_list:
        # # gaussian diffusion inverse
        # sigma_candidates = [0.3, 0.2, 0.15, 0.1, 0.05, 0.01]
        # lambda_candidates = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        # for sigma in sigma_candidates:
        #     for lam in lambda_candidates:
        #         cnn_subnetworks_evaluation_circle_rebuilt_cm(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                                      filtering_type={'residual_type': 'pseudoinverse'},
        #                                                      filtering_params={'sigma': sigma, 'lambda_reg': lam, 'reinforce': False},
        #                                                      selection_rate=selection_rate, feature_cm='pcc', 
        #                                                      subject_range=range(11,16), experiment_range=range(1,4), 
        #                                                      subnetworks_extract='read', subnetworks_exrtact_basis=range(1,11),
        #                                                      save=True)
        
        # graph tikhonov inverse
        alpha_candidates = [0.3, 0.2, 0.1, 0.05]
        lambda_candidates = [1e-4, 1e-3, 1e-2, 1e-1]
        for alpha in alpha_candidates:
            for lam in lambda_candidates:
                cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                         filtering_params={'computation': 'graph_tikhonov_inverse',
                                                                           'alpha': alpha, 'lambda': lam, 
                                                                           'normalized': False, 'reinforce': False},
                                                         selection_rate=selection_rate, feature_cm='pcc', 
                                                         apply_filter='graph_tikhonov_inverse',
                                                         subject_range=range(11,16), experiment_range=range(1,4),
                                                         subnetworks_extract='read', 
                                                         subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4), 
                                                         save=True)

def normal_evaluation_framework():
    # feature
    feature = 'pcc'
    
    # optimized parameters
    sigma, lamda = 0.1, 0.01
    
    # selection rates
    selection_rate_list = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    for selection_rate in selection_rate_list:
        # cnn_subnetworks_evaluation_circle_original_cm(selection_rate=selection_rate, feature_cm='pli', 
        #                                               subject_range=range(6,16), experiment_range=range(1,4), 
        #                                               subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
        #                                               save=True)
        
        # cnn_subnetworks_evaluation_circle_rebuilt_cm(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                              filter_params={'computation': 'pseudoinverse', 'lateral_mode': 'bilateral',
        #                                                             'sigma': sigma, 'lambda_reg': lamda, 'reinforce': False}, 
        #                                              selection_rate=selection_rate, feature_cm='pli', 
        #                                              subject_range=range(6,16), experiment_range=range(1,4), 
        #                                              subnetworks_extract = 'read', subnetworks_exrtact_basis=range(1, 6),
        #                                              save=True)
        
        # cnn_subnetworks_evaluation_circle_laplacian_cm(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                                filtering_params={'computation': 'laplacian', 'alpha': 0.1,
        #                                                                  'normalized': False, 'reinforce': False},
        #                                                selection_rate=selection_rate, feature_cm='pli', 
        #                                                subject_range=range(6,16), experiment_range=range(1,4), 
        #                                                subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
        #                                                save=True)
        
        cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                 filtering_params={'computation': 'graph_laplacian_denoising', 'cutoff_rank': 5,
                                                                   'normalized': False, 'reinforce': False},
                                                 selection_rate=1, feature_cm='pcc',
                                                 apply_filter='graph_laplacian_denoising',
                                                 subject_range=range(6,16), experiment_range=range(1,4),
                                                 subnetworks_extract='read',
                                                 subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                 save=True)

if __name__ == '__main__':
    # read, cal = global_average_consistency_check(apply_filter='diffusion_inverse',
    #                                              subnetworks_exrtact_basis=range(1,6))
    
    # parameters_optimization()
    
    normal_evaluation_framework()
    
    # %% End
    from cnn_val_circle import end_program_actions
    end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)