import numpy as np
import pytest
from scipy import sparse as sp

from moran_imaging.checks_balance import _astype_copy_false, _is_integral_float, check_consistent_length, is_multilabel


def test_astype_copy_false():
    # Test with dense array
    X_dense = np.array([[1.0, 2.0], [3.0, 4.0]])
    params_dense = _astype_copy_false(X_dense)
    assert params_dense == {"copy": False}

    # Test with sparse matrix (scipy version >= 1.1)
    X_sparse = sp.csr_matrix([[1.0, 0.0], [0.0, 4.0]])
    params_sparse = _astype_copy_false(X_sparse)
    assert params_sparse == {"copy": False}


def test_is_integral_float():
    # Test with integral float array
    y_integral = np.array([1.0, 2.0, 3.0])
    assert _is_integral_float(y_integral) is True

    # Test with non-integral float array
    y_non_integral = np.array([1.0, 2.5, 3.0])
    assert _is_integral_float(y_non_integral) is False

    # Test with integer array
    y_integer = np.array([1, 2, 3])
    assert _is_integral_float(y_integer) is False


def test_check_consistent_length():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    # Should not raise an error
    check_consistent_length(X, y)

    y_invalid = np.array([1, 2])
    with pytest.raises(ValueError):
        check_consistent_length(X, y_invalid)


def test_is_multilabel():
    # Test with multilabel indicator matrix
    y_multilabel = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    assert is_multilabel(y_multilabel) is True

    # Test with single label array
    y_singlelabel = np.array([1, 1, 1])
    assert is_multilabel(y_singlelabel) is False
