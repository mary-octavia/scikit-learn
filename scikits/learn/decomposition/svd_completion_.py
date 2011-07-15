"""SVD matrix completion"""

# Licence: BSD Style.

import numpy as np
import scipy.sparse as sp

def svd_completion(X, n_components, n_iter=100, k_w=1.0, k_h=1.0, mu=0.001):
    if not sp.issparse(X):
        X = np.atleast_2d(X)

    W = 1e-2 * np.random.randn(X.shape[0], n_components) 
    H = 1e-2 * np.random.randn(n_components, X.shape[1])
    rows, cols = X.nonzero()
    for ii in xrange(n_iter):
        for i, j in zip(rows, cols):
            p = np.dot(W[i, :], H[:, j])
            grad_W = (X[i, j] - p) * H[:, j] - k_w * W[i, :]
            grad_H = (X[i, j] - p) * W[i, :] - k_h * H[:, j]
            W[i, :] -= mu * grad_W
            H[:, j] -= mu * grad_H

    return W, H
