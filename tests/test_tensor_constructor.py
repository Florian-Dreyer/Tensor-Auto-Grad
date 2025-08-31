"""Tests for tensor constructor and validation."""

import numpy as np
import pytest

from tensor import Tensor


def test_constructor_basic():
    """Tests basic Tensor constructor functionality."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (2, 3))

    assert t.shape == (2, 3)
    assert t.dtype == np.float32
    assert t.requires_grad is True
    assert np.array_equal(t.data.reshape(t.shape), data)


def test_constructor_with_grads():
    """Tests Tensor constructor with pre-existing gradients."""
    data = np.array([1, 2, 3, 4], dtype=np.float32)  # Already flattened
    grads = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    t = Tensor(data, np.float32, True, (2, 2), _grads=grads)

    assert np.array_equal(t._grads.reshape(t.shape), grads.reshape(2, 2))


def test_constructor_with_children():
    """Tests Tensor constructor with children."""
    data = np.array([[1, 2]], dtype=np.float32)
    children = {Tensor(np.array([[0.5, 1.0]]), np.float32, True, (1, 2))}

    t = Tensor(data, np.float32, True, (1, 2), _children=children)

    assert t._children == children


def test_constructor_with_op():
    """Tests Tensor constructor with operation."""
    data = np.array([[1, 2]], dtype=np.float32)

    t = Tensor(data, np.float32, True, (1, 2), _op='+')

    assert t._op == '+'


def test_constructor_shape_validation():
    """Tests that constructor validates shape correctly."""
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)

    # Valid shape
    t = Tensor(data, np.float32, False, (2, 3))
    assert t.shape == (2, 3)

    # Invalid shape (wrong number of elements)
    with pytest.raises(ValueError, match='Shape of data and shape and not compatible'):
        Tensor(data, np.float32, False, (2, 2))  # 4 elements vs 6 elements

    # Valid shape (1D)
    t_1d = Tensor(data, np.float32, False, (6,))
    assert t_1d.shape == (6,)


def test_constructor_dtype_conversion():
    """Tests that constructor handles dtype conversion correctly."""
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)

    # Convert to float32
    t_float = Tensor(data, np.float32, False, (2, 2))
    assert t_float.dtype == np.float32
    assert np.array_equal(t_float.data.reshape(t_float.shape), data.astype(np.float32))

    # Keep as int32
    t_int = Tensor(data, np.int32, False, (2, 2))
    assert t_int.dtype == np.int32
    assert np.array_equal(t_int.data.reshape(t_int.shape), data)


def test_constructor_requires_grad():
    """Tests constructor with different requires_grad values."""
    data = np.array([[1, 2]], dtype=np.float32)

    # With requires_grad=True
    t_true = Tensor(data, np.float32, True, (1, 2))
    assert t_true.requires_grad is True

    # With requires_grad=False
    t_false = Tensor(data, np.float32, False, (1, 2))
    assert t_false.requires_grad is False


def test_constructor_empty_grads():
    """Tests constructor with empty gradients."""
    data = np.array([[1, 2]], dtype=np.float32)

    t = Tensor(data, np.float32, True, (1, 2), _grads=np.array([]))

    # Should initialize gradients to zeros
    assert np.allclose(t._grads, 0.0)


def test_constructor_empty_children():
    """Tests constructor with empty children."""
    data = np.array([[1, 2]], dtype=np.float32)

    t = Tensor(data, np.float32, True, (1, 2), _children=set())

    assert t._children == set()


def test_constructor_empty_op():
    """Tests constructor with empty operation."""
    data = np.array([[1, 2]], dtype=np.float32)

    t = Tensor(data, np.float32, True, (1, 2), _op='')

    assert t._op == ''


def test_constructor_3d_tensor():
    """Tests constructor with 3D tensor."""
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)

    t = Tensor(data, np.float32, False, (2, 2, 2))

    assert t.shape == (2, 2, 2)
    assert np.array_equal(t.data.reshape(t.shape), data)


def test_constructor_1d_tensor():
    """Tests constructor with 1D tensor."""
    data = np.array([1, 2, 3, 4], dtype=np.float32)

    t = Tensor(data, np.float32, False, (4,))

    assert t.shape == (4,)
    assert np.array_equal(t.data.reshape(t.shape), data)


def test_constructor_float64():
    """Tests constructor with float64 dtype."""
    data = np.array([[1.0, 2.0]], dtype=np.float64)

    t = Tensor(data, np.float64, False, (1, 2))

    assert t.dtype == np.float64
    assert np.array_equal(t.data.reshape(t.shape), data)


def test_constructor_int32():
    """Tests constructor with int32 dtype."""
    data = np.array([[1, 2]], dtype=np.int32)

    t = Tensor(data, np.int32, False, (1, 2))

    assert t.dtype == np.int32
    assert np.array_equal(t.data.reshape(t.shape), data)


def test_constructor_strides():
    """Tests that constructor computes strides correctly."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = Tensor(data, np.float32, False, (2, 3))

    # For shape (2, 3), strides should be (3, 1)
    expected_strides = (3, 1)
    assert t.strides == expected_strides


def test_constructor_strides_3d():
    """Tests that constructor computes strides correctly for 3D tensors."""
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    t = Tensor(data, np.float32, False, (2, 2, 2))

    # For shape (2, 2, 2), strides should be (4, 2, 1)
    expected_strides = (4, 2, 1)
    assert t.strides == expected_strides


def test_constructor_repr():
    """Tests that constructor creates proper string representation."""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (2, 2))

    repr_str = repr(t)

    # Check that repr contains expected information
    assert 'Tensor' in repr_str
    assert 'dtype=float32' in repr_str
    assert 'requires_grad=True' in repr_str
    assert 'shape=(2, 2)' in repr_str


def test_constructor_backward_function():
    """Tests that constructor initializes backward function."""
    data = np.array([[1, 2]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (1, 2))

    # Should have a default backward function (lambda: None)
    assert callable(t._backward)


def test_constructor_max_dim():
    """Tests that constructor sets max_dim correctly."""
    data = np.array([[1, 2]], dtype=np.float32)
    t = Tensor(data, np.float32, False, (1, 2))

    MAX_DIM = 2
    assert t._max_dim == MAX_DIM
