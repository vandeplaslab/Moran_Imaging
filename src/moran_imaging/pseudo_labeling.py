"""Pseudo labeling for the similarity matrix."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


# Define a function to make a symmetric matrix from a non-symmetric matrix
def make_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Make a symmetric matrix from a non-symmetric matrix."""
    # Use the maximum of the matrix and its transpose to make it symmetric
    symmetric_matrix = np.maximum(matrix, matrix.T)

    # Return the resulting symmetric matrix
    return symmetric_matrix


def run_knn(features: np.ndarray, k: int = 10) -> np.ndarray:
    """Run KNN on the features and return the adjacency matrix."""
    # Todo: Make a better solution for excluding self neighborhood
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(features)
    _, indices = nbrs.kneighbors(features)
    # Excluding self neighborhood here
    idx = indices[:, 1:]
    adj = np.zeros((features.shape[0], features.shape[0]))
    for i in range(len(idx)):
        adj[i, idx[i]] = 1
    return make_symmetric(adj)


def pseudo_labeling(
    ub: float, lb: float, sim: torch.tensor, index, knn: bool, knn_adj=None
) -> tuple[torch.tensor, torch.tensor]:
    """Pseudo labeling for the similarity matrix."""
    pos_loc = (sim >= ub).astype("float64")
    neg_loc = (sim <= lb).astype("float64")

    if knn:
        knn_submat = knn_adj[np.ix_(index, index)]
        # Todo: Not 100% sure with this one, should be checked again
        pos_loc = torch.tensor(np.maximum(pos_loc, knn_submat))
        neg_loc = torch.tensor(np.minimum(neg_loc, 1 - knn_submat))

    else:
        pos_loc = torch.tensor(pos_loc)
        neg_loc = torch.tensor(neg_loc)
    return pos_loc, neg_loc
