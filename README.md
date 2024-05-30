# PCA Implementation from Scratch

This repository contains an implementation of Principal Component Analysis (PCA) from scratch using Python. PCA is a statistical technique used to simplify a dataset by reducing its dimensionality while retaining most of the variance in the data.

## Description

PCA involves the following steps:
1. **Standardize the Data**: Ensure that each feature has a mean of 0 and a standard deviation of 1.
2. **Compute the Covariance Matrix**: Understand the relationships between different features.
3. **Compute the Eigenvalues and Eigenvectors**: Identify the principal components.
4. **Sort the Eigenvalues and Eigenvectors**: Sort them in descending order of the eigenvalues.
5. **Select Principal Components**: Choose the top k eigenvalues and their corresponding eigenvectors.
6. **Transform the Data**: Project the data onto the new subspace.

### Formulas

1. **Covariance Matrix**:
\[ \text{Cov}(X) = \frac{1}{n-1} (X - \bar{X})^T (X - \bar{X}) \]
where \( X \) is the data matrix, \( \bar{X} \) is the mean of the data matrix, and \( n \) is the number of observations.

2. **Eigenvalues and Eigenvectors**:
The eigenvalues \( \lambda \) and eigenvectors \( v \) satisfy:
\[ A v = \lambda v \]
where \( A \) is the covariance matrix.

## Code

The implementation includes the following steps:

1. **Loading the Dataset**: The Iris dataset is used as an example.
2. **Calculating the Covariance Matrix**.
3. **Computing Eigenvalues and Eigenvectors**.
4. **Sorting and Selecting Principal Components**.
5. **Transforming the Data**.

### pca.py

```python
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:27:31 2024

@author: DELL-ACXIOM 1
"""

import numpy as np
from sklearn.datasets import load_iris
from numpy import linalg

# Load the iris dataset
iris = load_iris()
X = iris['data']
y = iris['target']

def mean_funct(X):
    return np.mean(X, axis=0)  # Calculate the mean of each column

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

def PCA(X, components):
    # Calculate the covariance matrix
    cov_matrix = covariance_for_matrix(X)
    
    # Calculate eigenvalues and eigenvectors
    eig_vals, eig_vecs = linalg.eig(cov_matrix)
    
    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eig_vals)[::-1]
    sorted_eigenvalues = eig_vals[sorted_indices]
    sorted_eigenvectors = eig_vecs[:, sorted_indices]
    
    # Select the top 'components' eigenvectors
    principal_components = sorted_eigenvectors[:, :components]
    
    # Transform the data
    X_transformed = np.dot(X, principal_components)
    
    return X_transformed

# Perform PCA and transform the data
X_transformed = PCA(X, 2)

print("Transformed Data:")
print(X_transformed)
