import numpy as np

from moran_imaging.local_global_spatial_autocorrelation import GearyGlobal, MoranGlobal, MoranLocal
from moran_imaging.spatial_weights_matrix import define_spatial_weights_matrix


def _get_test_data():
    np.random.seed(0)
    image = np.random.randint(0, 100, size=(10, 10))
    weights = define_spatial_weights_matrix(
        image.shape, contiguity="queen", background_mask=None, with_lower_order=False
    )
    return image, weights


def test_moran_global():
    """Test MoranGlobal calculations."""
    ion_image, W = _get_test_data()
    obj = MoranGlobal(ion_image, W, [], num_permute=10)
    assert obj.Moran_I_global is not None
    assert obj.Moran_I_global > -1 and obj.Moran_I_global < 1
    assert obj.p_sim is not None
    assert obj.p_sim >= 0 and obj.p_sim <= 1


def test_geary_global():
    """Test GearyGlobal calculations."""
    ion_image, W = _get_test_data()
    obj = GearyGlobal(ion_image, W, num_permute=10)
    assert obj.Geary_c_global is not None
    assert obj.Geary_c_global > 0
    assert obj.p_sim is not None
    assert obj.p_sim >= 0 and obj.p_sim <= 1


def test_moran_local():
    """Test MoranLocal calculations."""
    ion_image, W = _get_test_data()
    obj = MoranLocal(ion_image, W, [], num_permute=10, significance_test=True)
    assert obj.Moran_I_local is not None
    assert len(obj.Moran_I_local) == ion_image.size
    assert obj.p_sim is not None
    assert (obj.p_sim >= 0).all() and (obj.p_sim <= 1).all()
