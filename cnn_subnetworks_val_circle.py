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

    # if file exsist
    if os.path.exists(file_path):
        try:
            # try to read sheet
            existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
        except ValueError:
            # if sheet not exsist then create empty DataFrame
            existing_df = pd.DataFrame()

        # concat by column
        df = pd.concat([existing_df, df], ignore_index=True)

        # continuation + replace
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        # if file not exsist then create
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
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

# %% cnn subnetworks evaluation circle common
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
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
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
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_origin.xlsx'
        sheet_name = f'sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

import spatial_gaussian_smoothing
def global_average_consistency_check(apply_filter='diffusion_inverse',
                                     selection_rate=1, feature_cm='pcc',
                                     subnetworks_exrtact_basis=range(1,6)):
    # projection
    projection_params={"source": "auto", "type": "3d_spherical"}
    
    # valid filters
    filters_valid={'gaussian_filtering', 
                   'diffusion_inverse', 'graph_laplacian_filtering', 
                   'generalized_surface_laplacian_filtering',
                   'truncated_graph_spectral_filtering', 'exp_graph_spectral_filtering',
                   'graph_tikhonov_inverse'}
    
    # default parameteres
    if apply_filter == 'gaussian_filtering':
        filtering_params={'computation': 'gaussian_filtering', 'lateral_mode': 'bilateral',
                          'sigma': 0.1, 'reinforce': False}
    elif apply_filter == 'diffusion_inverse':
        filtering_params={'computation': 'diffusion_inverse', 'lateral_mode': 'bilateral',
                          'sigma': 0.1, 'lambda_reg': 0.01, 'reinforce': False}    
    elif apply_filter == 'graph_laplacian_filtering':
        filtering_params={'computation': 'graph_laplacian_filtering', 'lateral_mode': 'bilateral',
                          'alpha': 1, 'sigma': 0.1, 'normalized': False, 'reinforce': False}
    
    elif apply_filter == 'truncated_graph_spectral_filtering':
        filtering_params={'computation': 'truncated_graph_spectral_filtering',
                          'cutoff': 0.1, 'mode': 'highpass', 'normalized': False, 'reinforce': False}
    elif apply_filter == 'exp_graph_spectral_filtering':
        filtering_params={'computation': 'exp_graph_spectral_filtering',
                          't': 5, 'mode': 'highpass', 'normalized': False, 'reinforce': False}
    
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
    filters_valid={'gaussian_filtering', 
                   'diffusion_inverse', 'graph_laplacian_filtering', 
                   'generalized_surface_laplacian_filtering',
                   'truncated_graph_spectral_filtering', 'exp_graph_spectral_filtering',
                   'graph_tikhonov_inverse'}
    
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
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
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
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:
        prj_type = projection_params.get('type')
        computation = filtering_params.get('computation')
        
        if apply_filter == 'diffusion_inverse':
            parameters_prompt = f'sigma_{filtering_params.get('sigma')}_lambda_reg_{filtering_params.get('lambda_reg')}'
        
        elif apply_filter == 'gaussian_filtering':
            parameters_prompt = f'sigma_{filtering_params.get('sigma')}'
        
        elif apply_filter == 'generalized_surface_laplacian_filtering':
            parameters_prompt = f'sigma_{filtering_params.get('sigma')}'
        
        elif apply_filter == 'graph_laplacian_filtering':
            parameters_prompt = f'mode_{filtering_params.get('mode')}_alpha_{filtering_params.get('alpha')}_sigma_{filtering_params.get('sigma')}'
        
        elif apply_filter == 'truncated_graph_spectral_filtering':
            parameters_prompt = f'mode_{filtering_params.get('mode')}_cutoff_{filtering_params.get('cutoff')}'
        
        elif apply_filter == 'exp_graph_spectral_filtering':
            parameters_prompt = f'mode_{filtering_params.get('mode')}_t_{filtering_params.get('t')}_sigma_{filtering_params.get('sigma')}'
        
        # elif apply_filter == 'graph_tikhonov_inverse':
        #     parameters_prompt = f'alpha_{filtering_params.get('alpha')}_lambda_{filtering_params.get('lambda')}'
        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{prj_type}_{computation}_{parameters_prompt}.xlsx'
        sheet_name = f'sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary (20251002)
        df_summary = pd.DataFrame([mean_row, std_row])
        save_to_xlsx_sheet(df_summary, folder_name, file_name, 'summary')
    
    return df_results

# %% Execute
def parameters_optimization():
    # selection_rate_list = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    
    selection_rate_list = [0.2, 0.1]
    
    for selection_rate in selection_rate_list:
        # gaussian diffusion inverse
        sigma_candidates = [0.3, 0.2, 0.15, 0.1, 0.05]
        lambda_candidates = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        lambda_candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        for sigma in sigma_candidates:
            for lam in lambda_candidates:               
                cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                         filtering_params={'computation': 'diffusion_inverse',
                                                                           'sigma': sigma, 'lambda_reg': lam,
                                                                           'lateral_mode': 'bilateral', 'reinforce': False},
                                                         selection_rate=selection_rate, feature_cm='pcc',
                                                         apply_filter='diffusion_inverse',
                                                         subject_range=range(6,16), experiment_range=range(1,4),
                                                         subnetworks_extract='read',
                                                         subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                         save=True)
        
        # # graph tikhonov inverse
        # alpha_candidates = [0.3, 0.2, 0.1, 0.05]
        # lambda_candidates = [1e-4, 1e-3, 1e-2, 1e-1]
        # for alpha in alpha_candidates:
        #     for lam in lambda_candidates:
        #         cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                                  filtering_params={'computation': 'graph_tikhonov_inverse',
        #                                                                    'alpha': alpha, 'lambda': lam, 
        #                                                                    'normalized': False, 'reinforce': False},
        #                                                  selection_rate=selection_rate, feature_cm='pcc', 
        #                                                  apply_filter='graph_tikhonov_inverse',
        #                                                  subject_range=range(11,16), experiment_range=range(1,4),
        #                                                  subnetworks_extract='read', 
        #                                                  subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4), 
        #                                                  save=True)

def normal_evaluation_framework():
    # feature
    feature = 'plv'
    
    # optimized parameters
    sigma, lamda = 0.1, 0.01
    
    # selection rates
    selection_rate_list = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    for selection_rate in selection_rate_list:
        #-----------------------------------------------------------------------
        # # original functional connectivity networks
        # cnn_subnetworks_evaluation_circle_original_cm(selection_rate=selection_rate, feature_cm=feature, 
        #                                               subject_range=range(6,16), experiment_range=range(1,4), 
        #                                               subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
        #                                               save=True)
        
        # # gaussian_filter; sigma = 0.1
        # cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                          filtering_params={'computation': 'gaussian_filtering',
        #                                                            'sigma': 0.1, 
        #                                                            'lateral_mode': 'bilateral', 'reinforce': False},
        #                                          selection_rate=selection_rate, feature_cm=feature,
        #                                          apply_filter='gaussian_filtering',
        #                                          subject_range=range(6,16), experiment_range=range(1,4),
        #                                          subnetworks_extract='read',
        #                                          subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                          save=True)
        
        #-----------------------------------------------------------------------
        # # diffusion_inverse; sigma, lambda = 0.1, 0.01
        # # optimized parameters: sigma = 0.1, lambda = 0.01
        # cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                          filtering_params={'computation': 'diffusion_inverse',
        #                                                            'sigma': 0.1, 'lambda_reg': 0.01,
        #                                                            'lateral_mode': 'bilateral', 'reinforce': False},
        #                                          selection_rate=selection_rate, feature_cm=feature,
        #                                          apply_filter='diffusion_inverse',
        #                                          subject_range=range(6,16), experiment_range=range(1,4),
        #                                          subnetworks_extract='read',
        #                                          subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                          save=True)
        
        #-----------------------------------------------------------------------
        # generalized_surface_laplacian_filtering; sigma = 0.01, 0.025, 0.05, ...; 'normalized': False
        # optimized parameter: sigma = 0.025
        # cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                          filtering_params={'computation': 'generalized_surface_laplacian_filtering', 
        #                                                            'sigma': 0.01, 
        #                                                            'normalized': False, 'reinforce': False},
        #                                          selection_rate=selection_rate, feature_cm=feature,
        #                                          apply_filter='generalized_surface_laplacian_filtering',
        #                                          subject_range=range(6,16), experiment_range=range(1,4),
        #                                          subnetworks_extract='read',
        #                                          subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                          save=True)
        
        #-----------------------------------------------------------------------
        # # graph_laplacian_filtering; highpass; alpha = 1; sigma = 0.05, 0.1, 0.25, ...; 'normalized': False
        # optimized parameter: sigma = 0.1
        # cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                          filtering_params={'computation': 'graph_laplacian_filtering',
        #                                                            'alpha': 1, 'sigma': 0.05,
        #                                                            'mode': 'highpass',
        #                                                            'lateral_mode': 'bilateral',
        #                                                            'normalized': False, 'reinforce': False},
        #                                          selection_rate=selection_rate, feature_cm=feature,
        #                                          apply_filter='graph_laplacian_filtering',
        #                                          subject_range=range(6,16), experiment_range=range(1,4),
        #                                          subnetworks_extract='read',
        #                                          subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                          save=True)
        
        #-----------------------------------------------------------------------
        # # graph_laplacian_filtering; lowpass; alpha = 1; sigma = 0.05, 0.1, 0.25, ...; 'normalized': False
        # optimized parameter: sigma = 0.1
        # cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
        #                                          filtering_params={'computation': 'graph_laplacian_filtering',
        #                                                            'alpha': 1, 'sigma': 0.05,
        #                                                            'mode': 'lowpass',
        #                                                            'lateral_mode': 'bilateral',
        #                                                            'normalized': False, 'reinforce': False},
        #                                          selection_rate=selection_rate, feature_cm=feature,
        #                                          apply_filter='graph_laplacian_filtering',
        #                                          subject_range=range(6,16), experiment_range=range(1,4),
        #                                          subnetworks_extract='read',
        #                                          subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
        #                                          save=True)
        
        #-----------------------------------------------------------------------
        # exp_graph_spectral_filtering; highpass; t = 1; sigma = 0 .05, 0.1, 0.25, 0.5 'normalized': False
        # optimized parameter: sigma = 0.1
        cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                 filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                                   't': 1, 'sigma': 0.05, 'mode': 'highpass',
                                                                   'normalized': False},
                                                 selection_rate=selection_rate, feature_cm=feature,
                                                 apply_filter='exp_graph_spectral_filtering',
                                                 subject_range=range(6,16), experiment_range=range(1,4),
                                                 subnetworks_extract='read', 
                                                 subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                 save=True)
        
        cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                 filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                                   't': 1, 'sigma': 0.1, 'mode': 'highpass',
                                                                   'normalized': False},
                                                 selection_rate=selection_rate, feature_cm=feature,
                                                 apply_filter='exp_graph_spectral_filtering',
                                                 subject_range=range(6,16), experiment_range=range(1,4),
                                                 subnetworks_extract='read', 
                                                 subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                 save=True)
        
        cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                 filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                                   't': 1, 'sigma': 0.25, 'mode': 'highpass',
                                                                   'normalized': False},
                                                 selection_rate=selection_rate, feature_cm=feature,
                                                 apply_filter='exp_graph_spectral_filtering',
                                                 subject_range=range(6,16), experiment_range=range(1,4),
                                                 subnetworks_extract='read', 
                                                 subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                 save=True)
        
        cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                 filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                                   't': 1, 'sigma': 0.5, 'mode': 'highpass',
                                                                   'normalized': False},
                                                 selection_rate=selection_rate, feature_cm=feature,
                                                 apply_filter='exp_graph_spectral_filtering',
                                                 subject_range=range(6,16), experiment_range=range(1,4),
                                                 subnetworks_extract='read', 
                                                 subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                 save=True)
        
        # exp_graph_spectral_filtering; lowpass; t = 1; sigma = 0 .05, 0.1, 0.25, 0.5 'normalized': False
        # optimized parameter: sigma = 0.01
        cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                 filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                                   't': 1, 'sigma': 0.01, 'mode': 'lowpass',
                                                                   'normalized': False},
                                                 selection_rate=selection_rate, feature_cm=feature,
                                                 apply_filter='exp_graph_spectral_filtering',
                                                 subject_range=range(6,16), experiment_range=range(1,4),
                                                 subnetworks_extract='read', 
                                                 subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                 save=True)
        
        cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                 filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                                   't': 1, 'sigma': 0.025, 'mode': 'lowpass',
                                                                   'normalized': False},
                                                 selection_rate=selection_rate, feature_cm=feature,
                                                 apply_filter='exp_graph_spectral_filtering',
                                                 subject_range=range(6,16), experiment_range=range(1,4),
                                                 subnetworks_extract='read', 
                                                 subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                 save=True)
        
        cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                 filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                                   't': 1, 'sigma': 0.05, 'mode': 'lowpass',
                                                                   'normalized': False},
                                                 selection_rate=selection_rate, feature_cm=feature,
                                                 apply_filter='exp_graph_spectral_filtering',
                                                 subject_range=range(6,16), experiment_range=range(1,4),
                                                 subnetworks_extract='read', 
                                                 subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                 save=True)
        
        cnn_subnetworks_evaluation_circle_common(projection_params={"source": "auto", "type": "3d_spherical"},
                                                 filtering_params={'computation': 'exp_graph_spectral_filtering', 
                                                                   't': 1, 'sigma': 0.1, 'mode': 'lowpass',
                                                                   'normalized': False},
                                                 selection_rate=selection_rate, feature_cm=feature,
                                                 apply_filter='exp_graph_spectral_filtering',
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
    end_program_actions(play_sound=True, shutdown=True, countdown_seconds=120)