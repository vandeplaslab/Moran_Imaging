"""Test fast_moran_felzenszwalb_segmentation.py"""

import numpy as np
import pytest

from moran_imaging.fast_moran_felzenszwalb_segmentation import get_moran_felzenszwalb_segmentation


@pytest.mark.parametrize("neighbourhood_order", [1, 2])
@pytest.mark.parametrize("neighbourhood_type", ["queen", "rook", "radial"])
@pytest.mark.parametrize("n_clusters", [2, 5, 10])
def test_get_moran_felzenszwalb_segmentation(n_clusters, neighbourhood_type, neighbourhood_order):
    image = np.random.rand(20, 10, 10)
    mask = np.ones((10, 10), dtype=bool)

    segments = get_moran_felzenszwalb_segmentation(
        image,
        mask,
        n_clusters=n_clusters,
        neighbourhood_type=neighbourhood_type,
        neighbourhood_order=neighbourhood_order,
    )

    assert segments is not None
    assert segments.shape == (10, 10)
    assert len(np.unique(segments)) <= n_clusters
