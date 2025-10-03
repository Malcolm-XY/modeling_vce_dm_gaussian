# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 22:45:31 2025

@author: 18307
"""

import numpy as np

def selection_robust_auc(srs, accuracies):
    aucs = []
    n = len(srs) - 1
    for i in range(n):
        auc = (srs[i]-srs[i+1]) * (accuracies[i]+accuracies[i+1])/2
        aucs.append(auc)
        
        print(i)
        print(auc)
        
    auc = np.sum(aucs) * 1/(srs[0]-srs[-1])
    
    return auc

srs = [1, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
accuracies = [93.10136478, 91.32666373, 89.72256104, 83.85720465, 76.24294905, 60.40277742, 47.7409003]

auc = selection_robust_auc(srs, accuracies)
print(auc)