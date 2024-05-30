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

2. **Eigenvalues and Eigenvectors
