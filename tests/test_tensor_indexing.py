"""Tests for tensor indexing operations: getitem, setitem, view."""

import numpy as np
import pytest

from tensor import Tensor


def test_tensor_getitem():
    """Tests Tensor indexing (__getitem__) with various indices."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = Tensor(data, np.float32, False, (2, 3))

    # Test single element access
    assert t[0, 0] == 1.0
    assert t[0, 1] == 2.0
    assert t[1, 2] == 6.0

    # Test single index (should be converted to tuple)
    assert t[0] == 1.0  # First element in flattened array


def test_tensor_setitem():
    """Tests Tensor assignment (__setitem__) with various indices."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = Tensor(data, np.float32, False, (2, 3))

    # Test setting single elements
    t[0, 0] = 10.0
    assert t[0, 0] == 10.0

    t[1, 2] = 20.0
    assert t[1, 2] == 20.0

    # Test setting with single index
    t[0] = 30.0
    assert t[0] == 30.0


def test_tensor_indexing_errors():
    """Tests that invalid indexing raises appropriate errors."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = Tensor(data, np.float32, False, (2, 3))

    # Test wrong number of indices
    with pytest.raises(IndexError):
        _ = t[0, 1, 2]  # Too many indices

    # Single index for 2D tensor now works for flattened access
    assert t[0] == 1.0  # First element in flattened array

    # Test out of bounds indices
    with pytest.raises(IndexError):
        _ = t[2, 0]  # Row index out of bounds

    with pytest.raises(IndexError):
        _ = t[0, 3]  # Column index out of bounds

    with pytest.raises(IndexError):
        _ = t[-1, 0]  # Negative index


def test_tensor_view():
    """Tests Tensor view method with various shapes."""
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (2, 4))

    # Test reshaping to different 2D shape
    t_view = t.view((4, 2))
    assert t_view.shape == (4, 2)
    assert t_view.data is t.data  # Same data reference
    assert t_view._grads is t._grads  # Same gradients reference

    # Test reshaping to 1D
    t_view_1d = t.view((8,))
    assert t_view_1d.shape == (8,)
    assert t_view_1d.data is t.data

    # Test reshaping back to original
    t_view_orig = t.view((2, 4))
    assert t_view_orig.shape == (2, 4)
    assert t_view_orig.data is t.data


def test_tensor_view_invalid_shape():
    """Tests that view with incompatible shape raises ValueError."""
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    t = Tensor(data, np.float32, False, (2, 4))

    # Test incompatible shape (wrong total elements)
    with pytest.raises(ValueError):
        _ = t.view((3, 3))  # 9 elements vs 8 elements

    with pytest.raises(ValueError):
        _ = t.view((2, 3))  # 6 elements vs 8 elements


def test_tensor_view_gradient_preservation():
    """Tests that view preserves gradients correctly."""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (2, 2))

    # Set some gradients
    t._grads = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    # Create view
    t_view = t.view((4,))

    # Check that gradients are preserved
    np.testing.assert_array_equal(t_view._grads, t._grads)

    # Modify gradients through view
    t_view._grads[0] = 1.0

    # Check that original tensor gradients are also modified
    assert t._grads[0] == 1.0


def test_tensor_indexing_with_gradients():
    """Tests that indexing works correctly with gradients."""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (2, 2))

    # Set gradients
    t._grads = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    # Test that indexing doesn't affect gradients
    value = t[1, 1]
    assert value == 4.0
    assert t._grads[3] == 0.4  # Gradient for element [1, 1]


def test_tensor_setitem_with_gradients():
    """Tests that setitem works correctly with gradients."""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (2, 2))

    # Set gradients
    t._grads = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    # Modify data
    t[0, 1] = 10.0

    # Check that data is modified but gradients remain
    assert t[0, 1] == 10.0
    assert t._grads[1] == 0.2  # Gradient for element [0, 1] should remain


def test_tensor_view_requires_grad():
    """Tests that view preserves requires_grad attribute."""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    t = Tensor(data, np.float32, True, (2, 2))

    t_view = t.view((4,))
    assert t_view.requires_grad == t.requires_grad

    # Test with requires_grad=False
    t_no_grad = Tensor(data, np.float32, False, (2, 2))
    t_no_grad_view = t_no_grad.view((4,))
    assert t_no_grad_view.requires_grad == t_no_grad.requires_grad
