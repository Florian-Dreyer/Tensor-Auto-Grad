"""Tests for tensor properties: shape, dtype, grad, requires_grad."""

import numpy as np
import pytest

from tensor import Tensor


def test_shape_property():
    """Tests Tensor shape property."""
    # Test 2D tensor
    data_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t_2d = Tensor(data_2d, np.float32, False, (2, 3))
    assert t_2d.shape == (2, 3)

    # Test 1D tensor
    data_1d = np.array([1, 2, 3, 4], dtype=np.float32)
    t_1d = Tensor(data_1d, np.float32, False, (4,))
    assert t_1d.shape == (4,)

    # Test 3D tensor
    data_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    t_3d = Tensor(data_3d, np.float32, False, (2, 2, 2))
    assert t_3d.shape == (2, 2, 2)


def test_shape_setter():
    """Tests Tensor shape setter."""
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    t = Tensor(data, np.float32, False, (2, 3))

    # Test valid shape change
    t.shape = (3, 2)
    assert t.shape == (3, 2)

    # Test invalid shape (wrong number of elements)
    with pytest.raises(ValueError):
        t.shape = (2, 2)  # 4 elements vs 6 elements

    # Test valid shape change back
    t.shape = (6,)
    assert t.shape == (6,)


def test_dtype_property():
    """Tests Tensor dtype property."""
    # Test float32
    data_float32 = np.array([[1.0, 2.0]], dtype=np.float32)
    t_float32 = Tensor(data_float32, np.float32, False, (1, 2))
    assert t_float32.dtype == np.float32

    # Test int32
    data_int32 = np.array([[1, 2]], dtype=np.int32)
    t_int32 = Tensor(data_int32, np.int32, False, (1, 2))
    assert t_int32.dtype == np.int32

    # Test float64
    data_float64 = np.array([[1.0, 2.0]], dtype=np.float64)
    t_float64 = Tensor(data_float64, np.float64, False, (1, 2))
    assert t_float64.dtype == np.float64


def test_grad_property():
    """Tests Tensor grad property."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (2, 2))

    # Initially gradients should be zero
    assert np.allclose(t.grad, 0.0)

    # Set some gradients
    t._grads = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    # Check grad property returns correct shape
    grad_reshaped = t.grad.reshape(t.shape)
    expected_grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    np.testing.assert_array_equal(grad_reshaped, expected_grad)


def test_requires_grad_property():
    """Tests Tensor requires_grad property."""
    data = np.array([[1.0, 2.0]], dtype=np.float32)

    # Test with requires_grad=True
    t_true = Tensor(data, np.float32, True, (1, 2))
    assert t_true.requires_grad is True

    # Test with requires_grad=False
    t_false = Tensor(data, np.float32, False, (1, 2))
    assert t_false.requires_grad is False


def test_properties_after_operations():
    """Tests that properties are preserved after operations."""
    a = Tensor(np.array([[1.0, 2.0]]), np.float32, True, (1, 2))
    b = Tensor(np.array([[0.5, 1.0]]), np.float32, True, (1, 2))

    # Test addition
    result = a + b
    assert result.shape == (1, 2)
    assert result.dtype == np.float32
    assert result.requires_grad is True

    # Test multiplication with scalar
    result_mul = a * 2.0
    assert result_mul.shape == (1, 2)
    # dtype might change due to numpy's type promotion rules
    assert result_mul.requires_grad is True


def test_properties_with_different_dtypes():
    """Tests properties with different data types."""
    # Test int32
    data_int = np.array([[1, 2]], dtype=np.int32)
    t_int = Tensor(data_int, np.int32, False, (1, 2))
    assert t_int.dtype == np.int32
    assert t_int.shape == (1, 2)

    # Test float64
    data_float = np.array([[1.0, 2.0]], dtype=np.float64)
    t_float = Tensor(data_float, np.float64, False, (1, 2))
    assert t_float.dtype == np.float64
    assert t_float.shape == (1, 2)

    # Test operation between different dtypes
    result = t_int + t_float
    # Result dtype should be the more general type (float64)
    assert result.dtype == np.float64


def test_grad_property_without_requires_grad():
    """Tests grad property when requires_grad=False."""
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    t = Tensor(data, np.float32, False, (1, 2))

    # Grad should still be accessible but zero
    assert np.allclose(t.grad, 0.0)

    # Setting gradients should still work
    t._grads = np.array([0.1, 0.2], dtype=np.float32)
    assert not np.allclose(t.grad, 0.0)


def test_properties_consistency():
    """Tests that properties are consistent across tensor operations."""
    a = Tensor(np.array([[1.0, 2.0]]), np.float32, True, (1, 2))
    b = Tensor(np.array([[0.5, 1.0]]), np.float32, True, (1, 2))

    # Test various operations
    operations = [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * 2.0,
        lambda x, y: x / 2.0,
        lambda x, y: x**2,
    ]

    for op in operations:
        result = op(a, b) if op.__code__.co_argcount == 2 else op(a)

        # Check that properties are consistent
        assert result.shape == (1, 2)
        # dtype might change due to numpy's type promotion rules
        assert result.requires_grad is True


def test_properties_with_view():
    """Tests that properties work correctly with view method."""
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (2, 4))

    # Create view
    t_view = t.view((4, 2))

    # Check properties
    assert t_view.shape == (4, 2)
    assert t_view.dtype == np.float32
    assert t_view.requires_grad is True

    # Check that data is shared
    assert t_view.data is t.data
    assert t_view._grads is t._grads


def test_properties_after_backward():
    """Tests that properties remain unchanged after backward pass."""
    a = Tensor(np.array([[1.0, 2.0]]), np.float32, True, (1, 2))
    b = Tensor(np.array([[0.5, 1.0]]), np.float32, True, (1, 2))

    result = a + b

    # Store original properties
    original_shape = result.shape
    original_dtype = result.dtype
    original_requires_grad = result.requires_grad

    # Perform backward pass
    result.backward()

    # Check that properties are unchanged
    assert result.shape == original_shape
    assert result.dtype == original_dtype
    assert result.requires_grad == original_requires_grad
