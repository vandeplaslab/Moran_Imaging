# Original code from https://github.com/hsmaan/balanced-clustering
# Paper: "Characterizing the impacts of dataset imbalance on single-cell data integration"
# by Maan, Zhang, Yu, Geuenich, Campbell and Wang, Nature Biotechnology, 2024.
# DOI: 10.1038/s41587-023-02097-9

from math import log

import numpy as np
import scipy.sparse as sp
from sklearn.utils import sparsefuncs

from moran_imaging.checks_balance import check_array, check_clusterings


def contingency_matrix(labels_true, labels_pred, *, reweigh=False, eps=None, sparse=False, dtype=np.int64):
    """Build a contingency matrix describing the relationship between labels.

    Parameters.
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.
    reweigh : bool, default=False
        if `True`, reweighs the contingency table based on the true labels
        such that they all have equal membership. The total number of samples
        is preserved with a round-off error.
    eps : float, default=None
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.
    sparse : bool, default=False
        If `True`, return a sparse CSR continency matrix. If `eps` is not
        `None` and `sparse` is `True` will raise ValueError.
        .. versionadded:: 0.18
    dtype : numeric type, default=np.int64
        Output dtype. Ignored if `eps` is not `None`.
        .. versionadded:: 0.24

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer unless set
        otherwise with the ``dtype`` argument. If ``eps`` is given, the dtype
        will be float.
        Will be a ``sklearn.sparse.csr_matrix`` if ``sparse=True``.
    """
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix(
        (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=dtype,
    )
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    # reweight contingency table if indicated
    if reweigh is True:
        contingency = contingency.astype(np.float64)
        counts_sum_per_class = np.ravel(contingency.sum(1))
        target = round(np.mean(counts_sum_per_class))
        counts_norm = counts_sum_per_class / target
        sparsefuncs.inplace_row_scale(contingency, 1 / counts_norm)
        contingency = contingency.astype(np.int64)

    return contingency


def pair_confusion_matrix(labels_true, labels_pred, reweigh=False):
    """Pair confusion matrix arising from two clusterings.
    The pair confusion matrix :math:`C` computes a 2 by 2 similarity matrix
    between two clusterings by considering all pairs of samples and counting
    pairs that are assigned into the same or into different clusters under
    the true and predicted clusterings.
    Considering a pair of samples that is clustered together a positive pair,
    then as in binary classification the count of true negatives is
    :math:`C_{00}`, false negatives is :math:`C_{10}`, true positives is
    :math:`C_{11}` and false positives is :math:`C_{01}`.
    Read more in the :ref:`User Guide <pair_confusion_matrix>`.

    Parameters.
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.
    reweigh : bool, default=False
        if `True`, reweighs the contingency table based on the true labels
        such that they all have equal membership. The total number of samples
        is preserved with a round-off error.

    Returns
    -------
    C : ndarray of shape (2, 2), dtype=np.int64
        The contingency matrix.
    ------
    Note that the matrix is not symmetric.
    ------

    References
    ----------
    .. L. Hubert and P. Arabie, Comparing Partitions, Journal of
      Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075
    """
    n_samples = np.int64(labels_true.shape[0])

    # Computation using the contingency data
    contingency = contingency_matrix(labels_true, labels_pred, reweigh=reweigh, sparse=True, dtype=np.int64)
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency.data**2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares
    return C


def entropy(labels):
    """Calculates the entropy for a labeling.

    Parameters.
    ----------
    labels : int array, shape = [n_samples]
        The labels

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))


def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    r"""Mutual Information between two clusterings.
    The Mutual Information is a measure of the similarity between two labels
    of the same data. Where :math:`|U_i|` is the number of the samples
    in cluster :math:`U_i` and :math:`|V_j|` is the number of the
    samples in cluster :math:`V_j`, the Mutual Information
    between clusterings :math:`U` and :math:`V` is given as:
    .. math::
        MI(U,V)=\\sum_{i=1}^{|U|} \\sum_{j=1}^{|V|} \\frac{|U_i\\cap V_j|}{N}
        \\log\\frac{N|U_i \\cap V_j|}{|U_i||V_j|}
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching :math:`U` (i.e
    ``label_true``) with :math:`V` (i.e. ``label_pred``) will return the
    same score value. This can be useful to measure the agreement of two
    independent label assignments strategies on the same dataset when the
    real ground truth is not known.
    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters.
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.
    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.
    contingency : {ndarray, sparse matrix} of shape \
            (n_classes_true, n_classes_pred), default=None
        A contingency matrix given by the :func:`contingency_matrix` function.
        If value is ``None``, it will be computed, otherwise the given value is
        used, with ``labels_true`` and ``labels_pred`` ignored.

    Returns
    -------
    mi : float
       Mutual information, a non-negative value, measured in nats using the
       natural logarithm.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    See Also
    --------
    adjusted_mutual_info_score : Adjusted against chance Mutual Information.
    normalized_mutual_info_score : Normalized Mutual Information.
    """
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    else:
        contingency = check_array(
            contingency,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[int, np.int32, np.int64],
        )

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    elif sp.issparse(contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)
    else:
        raise ValueError("Unsupported type for 'contingency': %s" % type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(np.int64, copy=False)
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = contingency_nm * (log_contingency_nm - log(contingency_sum)) + contingency_nm * log_outer
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)
