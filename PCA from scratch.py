# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:27:31 2024

@author: DELL-ACXIOM 1
"""

import numpy as np
from sklearn.datasets import load_iris
from numpy import linalg

iris = load_iris()
X = iris['data']
y = iris['target']

def mean_funct(X):
    return np.mean(X)

def covariance(X, Y):
    x_mean = mean_funct(X)
    y_mean = mean_funct(Y)
    
    return np.sum((X - x_mean) * (Y - y_mean)) / (len(X) - 1)

def covariance_for_matrix(data):
    num_features = data.shape[1]
    cvm = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            cvm[i][j] = covariance(data[:, i], data[:, j])
    return cvm


def PCA(X,components):
    cov_matrix = covariance_for_matrix(X)

    eig_vals, eig_vecs = linalg.eig(cov_matrix)
    principal_values = eig_vals[:components]
    principal_vecs = eig_vecs[:components]

    sorted_indices = np.argsort(eig_vals)[::-1]

    sorted_eigenvalues = eig_vals[sorted_indices]
    sorted_eigenvectors = eig_vecs[:, sorted_indices]

    principal_components = sorted_eigenvectors[:,:components]
    X_transformed = np.dot(X, principal_components)

    return X_transformed


PCA(X,2)
