"""Tests for tensor utility functions: compute_strides, _flatten_index, _set_data."""

import numpy as np
import pytest

from tensor import tensor


def test_compute_strides():
    """Tests compute_strides static method."""
    # Test 1D tensor
    strides_1d = tensor.Tensor.compute_strides((5,))
    assert strides_1d == (1,)

    # Test 2D tensor
    strides_2d = tensor.Tensor.compute_strides((2, 3))
    assert strides_2d == (3, 1)

    # Test 3D tensor
    strides_3d = tensor.Tensor.compute_strides((2, 3, 4))
    assert strides_3d == (12, 4, 1)

    # Test 4D tensor
    strides_4d = tensor.Tensor.compute_strides((2, 3, 4, 5))
    assert strides_4d == (60, 20, 5, 1)


def test_compute_strides_edge_cases():
    """Tests compute_strides with edge cases."""
    # Test single element
    strides_single = tensor.Tensor.compute_strides((1,))
    assert strides_single == (1,)

    # Test empty shape (should not happen in practice)
    # This actually works with the current implementation
    strides_empty = tensor.Tensor.compute_strides(())
    assert strides_empty == ()


def test_flatten_index():
    """Tests _flatten_index method."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = tensor.Tensor(data, np.float32, False, (2, 3))

    # Test valid indices
    assert t._flatten_index((0, 0)) == 0
    assert t._flatten_index((0, 1)) == 1
    assert t._flatten_index((0, 2)) == 2
    assert t._flatten_index((1, 0)) == 3
    assert t._flatten_index((1, 1)) == 4
    assert t._flatten_index((1, 2)) == 5


def test_flatten_index_3d():
    """Tests _flatten_index with 3D tensor."""
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    t = tensor.Tensor(data, np.float32, False, (2, 2, 2))

    # Test valid indices
    assert t._flatten_index((0, 0, 0)) == 0
    assert t._flatten_index((0, 0, 1)) == 1
    assert t._flatten_index((0, 1, 0)) == 2
    assert t._flatten_index((0, 1, 1)) == 3
    assert t._flatten_index((1, 0, 0)) == 4
    assert t._flatten_index((1, 0, 1)) == 5
    assert t._flatten_index((1, 1, 0)) == 6
    assert t._flatten_index((1, 1, 1)) == 7


def test_flatten_index_errors():
    """Tests _flatten_index error conditions."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = tensor.Tensor(data, np.float32, False, (2, 3))

    # Test wrong number of indices
    with pytest.raises(IndexError):
        t._flatten_index((0, 1, 2))  # Too many indices

    # Test single index (should work now)
    assert t._flatten_index((0,)) == 0

    # Test out of bounds indices
    with pytest.raises(IndexError):
        t._flatten_index((2, 0))  # Row index out of bounds

    with pytest.raises(IndexError):
        t._flatten_index((0, 3))  # Column index out of bounds

    with pytest.raises(IndexError):
        t._flatten_index((-1, 0))  # Negative index


def test_set_data_basic():
    """Tests _set_data method with basic functionality."""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    shape = (2, 2)
    dtype = np.float32
    grads = np.array([])

    # Create a temporary tensor to test _set_data
    temp_tensor = tensor.Tensor(data, dtype, False, shape)
    result = temp_tensor._set_data(data, dtype, shape, grads)
    expected = np.array([1, 2, 3, 4], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_set_data_with_grads():
    """Tests _set_data method with existing gradients."""
    data = np.array([1, 2, 3, 4], dtype=np.float32)  # Already flattened
    shape = (2, 2)
    dtype = np.float32
    grads = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    # Create a temporary tensor to test _set_data
    temp_tensor = tensor.Tensor(data, dtype, False, shape)
    result = temp_tensor._set_data(data, dtype, shape, grads)
    expected = np.array([1, 2, 3, 4], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_set_data_dtype_conversion():
    """Tests _set_data method with dtype conversion."""
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    shape = (2, 2)
    dtype = np.float32
    grads = np.array([])

    temp_tensor = tensor.Tensor(data, dtype, False, shape)
    result = temp_tensor._set_data(data, dtype, shape, grads)
    expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_set_data_shape_validation():
    """Tests _set_data method shape validation."""
    data = np.array([1, 2, 3, 4], dtype=np.float32)
    shape = (2, 3)  # 6 elements vs 4 elements
    dtype = np.float32
    grads = np.array([])

    temp_tensor = tensor.Tensor(data, dtype, False, (4,))
    with pytest.raises(ValueError):
        temp_tensor._set_data(data, dtype, shape, grads)


def test_set_data_3d():
    """Tests _set_data method with 3D data."""
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    shape = (2, 2, 2)
    dtype = np.float32
    grads = np.array([])

    temp_tensor = tensor.Tensor(data, dtype, False, shape)
    result = temp_tensor._set_data(data, dtype, shape, grads)
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_set_data_1d():
    """Tests _set_data method with 1D data."""
    data = np.array([1, 2, 3, 4], dtype=np.float32)
    shape = (4,)
    dtype = np.float32
    grads = np.array([])

    temp_tensor = tensor.Tensor(data, dtype, False, shape)
    result = temp_tensor._set_data(data, dtype, shape, grads)
    expected = np.array([1, 2, 3, 4], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_set_data_with_none_grads():
    """Tests _set_data method with None gradients."""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    shape = (2, 2)
    dtype = np.float32
    grads = None

    temp_tensor = tensor.Tensor(data, dtype, False, shape)
    result = temp_tensor._set_data(data, dtype, shape, grads)
    expected = np.array([1, 2, 3, 4], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_strides_calculation_verification():
    """Tests that strides calculation is mathematically correct."""
    # Test with a 2x3x4 tensor
    shape = (2, 3, 4)
    strides = tensor.Tensor.compute_strides(shape)

    # Manually verify strides
    # For shape (2, 3, 4):
    # - Last dimension (4): stride = 1
    # - Second-to-last dimension (3): stride = 1 * 4 = 4
    # - First dimension (2): stride = 4 * 3 = 12
    expected_strides = (12, 4, 1)
    assert strides == expected_strides

    # Verify that strides work correctly for indexing
    data = np.arange(24).reshape(2, 3, 4)
    t = tensor.Tensor(data, np.float32, False, (2, 3, 4))

    # Test a few indices
    assert t._flatten_index((0, 0, 0)) == 0
    assert t._flatten_index((0, 0, 1)) == 1
    assert t._flatten_index((0, 1, 0)) == 4
    assert t._flatten_index((1, 0, 0)) == 12
    assert t._flatten_index((1, 2, 3)) == 23  # Last element


def test_flatten_index_consistency():
    """Tests that flatten_index is consistent with numpy's ravel_multi_index."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = tensor.Tensor(data, np.float32, False, (2, 3))

    # Test several indices
    test_indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for idx in test_indices:
        custom_result = t._flatten_index(idx)
        numpy_result = np.ravel_multi_index(idx, t.shape)
        assert custom_result == numpy_result


def test_set_data_preserves_data():
    """Tests that _set_data preserves the original data correctly."""
    original_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    shape = (2, 2)
    dtype = np.float32
    grads = np.array([])

    temp_tensor = tensor.Tensor(original_data, dtype, False, shape)
    flattened_data = temp_tensor._set_data(original_data, dtype, shape, grads)

    # Reshape back and compare
    reshaped_data = flattened_data.reshape(shape)
    np.testing.assert_array_equal(reshaped_data, original_data)
