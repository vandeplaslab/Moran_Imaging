"""Test spatial weights matrix definitions."""

import pytest

from moran_imaging.spatial_weights_matrix import define_spatial_weights_matrix


@pytest.mark.parametrize("contiguity", ["queen", "Queen", "rook", "Rook"])
@pytest.mark.parametrize("with_lower_order", [True, False])
@pytest.mark.parametrize("neighbourhood_order", [0, 1, 2])
def test_define_spatial_weights_matrix(contiguity, with_lower_order, neighbourhood_order):
    weights = define_spatial_weights_matrix(
        (10, 10), contiguity=contiguity, with_lower_order=with_lower_order, neighbourhood_order=neighbourhood_order
    )
    assert weights is not None, "Expected an object"
    assert not weights._cache, "Cache should be empty"
    assert weights.transform == "O", "Expected 'O' transform"
    assert weights.n_components != 0, "Expected at least 1 component"
