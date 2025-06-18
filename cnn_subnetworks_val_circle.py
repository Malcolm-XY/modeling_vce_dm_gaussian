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

import cw_manager

def cnn_subnetworks_evaluation_circle_control_1(argument='data_driven_pcc_10_15', selection_rate=1, feature_cm='pcc',
                                                subject_range=range(11,16), experiment_range=range(1,4), 
                                                save=False):
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
        
    channel_weights = cw_manager.read_channel_weight_DD(argument, sort=True)
    channel_selected = channel_weights.index[:int(len(channel_weights.index)*selection_rate)]
    
    # data and evaluation circle
    all_results_list = []

    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")

            # CM/MAT
            features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            alpha = features['alpha'][:,channel_selected,:][:,:,channel_selected]
            beta = features['beta'][:,channel_selected,:][:,:,channel_selected]
            gamma = features['gamma'][:,channel_selected,:][:,:,channel_selected]
            
            # CM/H5
            # features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            # alpha = features['alpha'][:,channel_selected,:][:,:,channel_selected]
            # beta = features['beta'][:,channel_selected,:][:,:,channel_selected]
            # gamma = features['gamma'][:,channel_selected,:][:,:,channel_selected]
            
            x = np.stack((alpha, beta, gamma), axis=1)

            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_CM = cnn_validation.cnn_cross_validation(cnn_model, x, y)

            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_CM}
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
        file_name = 'cnn_validation_SubCM_pcc_by_DDPCC.xlsx'
        file_name = f'cnn_validation_SubCM_{feature_cm}_by_DD{feature_cm.upper()}.xlsx'
        sheet_name = f'selection_rate_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

def cnn_subnetworks_evaluation_circle_control_2(argument='label_driven_mi_10_15', selection_rate=1, feature_cm='pcc',
                                                subject_range=range(11,16), experiment_range=range(1,4), 
                                                save=False, iden='mi'):
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
        
    channel_weights = cw_manager.read_channel_weight_LD(argument, sort=True)
    channel_selected = channel_weights.index[:int(len(channel_weights.index)*selection_rate)]
    
    # data and evaluation circle
    all_results_list = []

    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")

            # CM/MAT
            features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            alpha = features['alpha'][:,channel_selected,:][:,:,channel_selected]
            beta = features['beta'][:,channel_selected,:][:,:,channel_selected]
            gamma = features['gamma'][:,channel_selected,:][:,:,channel_selected]
            
            # CM/H5
            # features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            # alpha = features['alpha'][:,channel_selected,:][:,:,channel_selected]
            # beta = features['beta'][:,channel_selected,:][:,:,channel_selected]
            # gamma = features['gamma'][:,channel_selected,:][:,:,channel_selected]
            
            x = np.stack((alpha, beta, gamma), axis=1)

            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_CM = cnn_validation.cnn_cross_validation(cnn_model, x, y)

            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_CM}
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
        file_name = f'cnn_validation_SubCM_{feature_cm}_by_LD{iden.upper()}.xlsx'
        sheet_name = f'selection_rate_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

import spatial_gaussian_smoothing
def cnn_subnetworks_evaluation_circle_rebuilt_cm(projection_params={"source": "auto", "type": "3d"},
                                                 filtering_type={'residual_type': 'origin', 'lateral_mode': 'bilateral'},
                                                 filtering_params={'sigma': 0.125, 'gamma': 0.25, 'lambda_reg': 1e-3},
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
    
    # experiment; channel weights computed from the rebuilded connectivity matrix that constructed by vce modeling
    # global channel_weights
    # channel_weights = functional_node_strength['beta'] + functional_node_strength['alpha'] + functional_node_strength['gamma']
    channel_weights = functional_node_strength['gamma']
    channel_weights = np.mean(np.mean(channel_weights, axis=0), axis=0)
    k = int(len(channel_weights) * selection_rate)
    channel_selected = np.argsort(channel_weights)[-k:][::-1]
    
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
            
            # subnetworks
            alpha_rebuilded = alpha_rebuilded[:,channel_selected,:][:,:,channel_selected]
            beta_rebuilded = beta_rebuilded[:,channel_selected,:][:,:,channel_selected]
            gamma_rebuilded = gamma_rebuilded[:,channel_selected,:][:,:,channel_selected]
            
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
        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{prj_type}_prj_{residual_type}_kernel.xlsx'
        sheet_name = f'{residual_type}_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

def cnn_subnetworks_eval_circle_rcm_intergrated(projection_params, filtering_params, selection_rate, feature_cm, save=False):
    residual_list = list(['origin', 'origin_gaussian', 'inverse', 'residual_mean', 'pseudoinverse'])
    
    results_fitting = {}
    for trail in range(0, len(residual_list)):
        results_fitting[residual_list[trail]] = cnn_subnetworks_evaluation_circle_rebuilt_cm(projection_params,
                                                                                             filtering_type={'residual_type': residual_list[trail]},
                                                                                             filtering_params=filtering_params,
                                                                                             selection_rate=selection_rate, feature_cm=feature_cm,
                                                                                             save=save)
    
    return results_fitting

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

# %% Execute
if __name__ == '__main__':
    # selection_rate_list = [1, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    selection_rate_list = [1, 0.5, 0.3, 0.2, 0.1]
    
    for selection_rate in selection_rate_list:
        # cnn_subnetworks_evaluation_circle_control_1(argument='data_driven_pcc_10_15', selection_rate=selection_rate, feature_cm='pcc', save=True)
        # cnn_subnetworks_evaluation_circle_control_2(selection_rate=selection_rate, feature_cm='pcc', save=True)

        # cnn_subnetworks_eval_circle_rcm_intergrated(projection_params={"source": "auto", "type": "3d"},
        #                                             filtering_params={'sigma': 0.1, 'gamma': 0.1, 'lambda_reg': 0.25},
        #                                             selection_rate=selection_rate, feature_cm='pcc', save=True)
        
        cnn_subnetworks_eval_circle_rcm_intergrated(projection_params={"source": "manual", "type": "2d"},
                                                    filtering_params={'sigma': 0.1, 'gamma': 0.1, 'lambda_reg': 0.25},
                                                    selection_rate=selection_rate, feature_cm='pcc', save=True)
        
    # %% End
    from cnn_val_circle import end_program_actions
    end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)