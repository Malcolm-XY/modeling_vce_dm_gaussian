# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 19:09:12 2025

@author: 18307
"""

from utils import utils_feature_loading

de_gamma_sample = utils_feature_loading.read_cfs('seed', 'sub1ex1', 'de')['gamma']

electrodes = utils_feature_loading.read_distribution('seed')['channel']

from utils import utils_visualization

utils_visualization.draw_heatmap_1d(de_gamma_sample[0:99, 0:23].T, electrodes[0:23], figsize=(20,5))

# for i in range(0,10):
#     utils_visualization.draw_heatmap_1d(de_gamma_sample[i:i+1, 0:23].T, electrodes[0:23], [i], figsize=(0.2,5))

pcc_gamma_sample = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')['gamma']

utils_visualization.draw_projection(pcc_gamma_sample[0, 0:23, 0:23], None, electrodes[0:23], electrodes[0:23])

import feature_engineering

feature_engineering.compute_distance_matrix('seed', {'source': 'auto', 'type': '2d_flat', 'resolution': 37}, True)
feature_engineering.compute_distance_matrix('seed', {'source': 'manual', 'type': '2d_flat', 'resolution': 19}, True)