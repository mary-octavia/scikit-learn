import numpy as np

from sklearn.utils.hungarian import hungarian
from sklearn.utils.validation import check_arrays


def _check_rows_and_columns(a, b):
    a_rows, a_cols = check_arrays(*a)
    b_rows, b_cols = check_arrays(*b)
    return a_rows, a_cols, b_rows, b_cols


def _jaccard(a_rows, a_cols, b_rows, b_cols):
    intersection = ((a_rows * b_rows).sum() *
                    (a_cols * b_cols).sum())

    a_size = a_rows.sum() * a_cols.sum()
    b_size = b_rows.sum() * b_cols.sum()

    return intersection / (a_size + b_size - intersection)


def _pairwise_similarity(a, b):
    a_rows, a_cols, b_rows, b_cols = _check_rows_and_columns(a, b)
    n_a = a_rows.shape[0]
    n_b = b_rows.shape[0]
    result = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            result[i, j] = _jaccard(a_rows[i], a_cols[i], b_rows[j], b_cols[j])
    return result


def score_biclusters(a, b):
    """The similarity of two sets of biclusters.

    Similarity between individual biclusters is computed using the
    Jaccard index. Then the best matching between sets is found using
    the Hungarian algorithm. The final score is the sum of
    similarities divided by the size of the larger set.

    """
    matrix = _pairwise_similarity(a, b)
    indices = hungarian(1 - matrix)
    n_a = a[0].shape[0]
    n_b = b[0].shape[0]
    return np.trace(matrix[:, indices[:, 1]]).sum() / max(n_a, n_b)
