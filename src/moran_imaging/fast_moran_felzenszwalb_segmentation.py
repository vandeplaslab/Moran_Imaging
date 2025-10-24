# Parallelized implementation of the Moran-HOG segmentation workflow
# Refer to Section 2.2 of the Supplementary Material
# By Felipe Moser

import numpy as np
import scipy.ndimage
import scipy.stats
import skimage.segmentation
import sklearn.cluster


def get_local_moran_i(image, mask, neighbourhood_type="queen", neighbourhood_order=5):
    """
    Compute the local Moran I statistic for each pixel in an image.

    Parameters
    ----------
    image : np.ndarray of shape (n_channels, height, width)
        The image to segment

    mask : np.ndarray of shape (height, width)
        A boolean mask indicating the pixels to consider in the analysis

    neighbour_type [optional] : str
        The type of neighbourhood to consider. Either "Queen" or "Rook". Default is "Queen"

    neighbour_order [optional] : int
        The order of the neighbourhood. Default is 5

    Returns
    -------
    local_moran_i : np.ndarray of shape (n_channels, height, width)
        The local Moran I statistic for each pixel in the image

    quadrant_labels : np.ndarray of shape (n_channels, height, width)
        The quadrant label for each pixel in the image
    """
    # Get the local Moran I
    neighbourhood_kernel = get_neighbourhood_kernel(neighbourhood_type, neighbourhood_order)
    neighbourhood_kernel = np.tile(neighbourhood_kernel, (image.shape[0], 1, 1))
    z = image - np.mean(image[:, mask], axis=1)[:, None, None]
    z_lag = scipy.ndimage.convolve(z, neighbourhood_kernel, mode="constant", cval=0.0)
    numerator = z * z_lag
    denominator = ((z[:, mask > 0]) ** 2).sum(axis=(1))
    local_moran_i = numerator / denominator[:, None, None]
    local_moran_i *= mask

    # Assign each pixel to a quadrant of the Moran scatterplot
    z_pos = z > 0
    z_neg = np.invert(z_pos)
    lag_pos = z_lag > 0
    lag_neg = np.invert(lag_pos)
    pos_pos = np.logical_and(z_pos, lag_pos)
    neg_pos = np.logical_and(z_neg, lag_pos)
    neg_neg = np.logical_and(z_neg, lag_neg)
    pos_neg = np.logical_and(z_pos, lag_neg)
    quadrant_labels = pos_pos * 1.0 + neg_pos * 2.0 + neg_neg * 3.0 + pos_neg * 4.0
    quadrant_labels *= mask

    return local_moran_i, quadrant_labels


def get_moran_felzenszwalb_segmentation(
    image,
    mask,
    n_clusters,
    neighbourhood_type="queen",
    neighbourhood_order=5,
    felzenszwalb_scale=50,
    felzenszwalb_sigma=0.2,
    felzenszwalb_min_size=100,
    subset_features=None,
):
    """
    Segment an image using the Moran I statistic and the Felzenszwalb algorithm.

    Parameters
    ----------
    image : np.ndarray of shape (n_channels, height, width)
        The image to segment

    mask : np.ndarray of shape (height, width)
        A boolean mask indicating the pixels to consider in the analysis

    neighbour_type [optional] : str
        The type of neighbourhood to consider. Either "Queen" or "Rook". Default is "Queen"

    neighbour_order [optional] : int
        The order of the neighbourhood. Default is 5
    """
    if subset_features is None:
        subset_features = []
    if not subset_features:
        # Get the local Moran I and quadrant labels for all m/z bins
        _, quadrant_labels = get_local_moran_i(image, mask, neighbourhood_type, neighbourhood_order)
    else:
        # Get the local Moran I and quadrant labels for a subset of m/z bins
        _, quadrant_labels = get_local_moran_i(image[subset_features, :], mask, neighbourhood_type, neighbourhood_order)

    # Segment the image using the Felzenszwalb algorithm on the quadrant labels
    felzenszwalb_segmentation = skimage.segmentation.felzenszwalb(
        quadrant_labels,
        scale=felzenszwalb_scale,
        sigma=felzenszwalb_sigma,
        min_size=felzenszwalb_min_size,
        channel_axis=0,
    )

    # k-means clustering
    cluster_model = sklearn.cluster.KMeans(n_clusters=n_clusters, init="k-means++", n_init=50)
    cluster_model.fit(image[:, mask].T)

    # cluster_labels = cluster_model.labels_
    cluster_labels = np.zeros_like(mask, dtype=int)
    cluster_labels[mask] = cluster_model.labels_ + 1  # +1 to avoid 0 labels

    moran_felzenszwalb_segmentation = np.zeros_like(mask, dtype=int)
    for i in np.unique(felzenszwalb_segmentation):
        mask_segment = felzenszwalb_segmentation == i
        moran_felzenszwalb_segmentation[mask_segment] = scipy.stats.mode(cluster_labels[mask_segment])[0]

    return moran_felzenszwalb_segmentation


def get_neighbourhood_kernel(neighbourhood_type, neighbourhood_order):
    """
    Get a neighbourhood kernel for a given neighbourhood type and order.

    Parameters
    ----------
    neighbourhood_type : str
        The type of neighbourhood to consider. Either "Queen", "Rook" or "Radial"

    neighbourhood_order : int
        The order of the neighbourhood

    Returns
    -------
    w : np.ndarray of shape (2 * neighbourhood_order + 1, 2 * neighbourhood_order + 1)
        The neighbourhood kernel
    """
    if neighbourhood_type.lower() == "queen":
        w = np.ones((2 * neighbourhood_order + 1, 2 * neighbourhood_order + 1))
        w[neighbourhood_order, neighbourhood_order] = 0
        w /= w.sum()

    elif neighbourhood_type.lower() == "rook":
        Y, X = np.ogrid[: 2 * neighbourhood_order + 1, : 2 * neighbourhood_order + 1]
        w = np.abs(X - neighbourhood_order) + np.abs(Y - neighbourhood_order) <= neighbourhood_order
        w[neighbourhood_order, neighbourhood_order] = 0
        w = w.astype(float)
        w /= w.sum()

    elif neighbourhood_type.lower() == "radial":
        Y, X = np.ogrid[: 2 * neighbourhood_order + 1, : 2 * neighbourhood_order + 1]
        w = (X - neighbourhood_order) ** 2 + (Y - neighbourhood_order) ** 2 <= neighbourhood_order**2
        w[neighbourhood_order, neighbourhood_order] = 0
        w = w.astype(float)
        w /= w.sum()
    return w
