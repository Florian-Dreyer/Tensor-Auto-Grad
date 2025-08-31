"""Tests for tensor operations: add, sub, mul, div, pow, matmul."""

import numpy as np
import pytest
import torch

from tensor import tensor


def _assert_tensor_equals(
    custom_tensor: tensor.Tensor, torch_tensor: torch.Tensor, check_grad: bool = False
):
    """Compares custom Tensor data and grads (if requested) with a PyTorch tensor."""
    # data
    custom_data = custom_tensor.data.reshape(custom_tensor.shape).astype(np.float32)
    torch_data = torch_tensor.detach().numpy()
    np.testing.assert_allclose(
        custom_data, torch_data, rtol=1e-5, atol=1e-5, err_msg='Data mismatch'
    )
    # shape
    assert custom_tensor.shape == tuple(torch_tensor.shape), 'Shape mismatch'
    # Grad
    assert (
        custom_tensor.requires_grad == torch_tensor.requires_grad
    ), 'requires_grad mismatch'
    if check_grad and custom_tensor.requires_grad and torch_tensor.requires_grad:
        assert torch_tensor.grad is not None, 'PyTorch tensor gradient is None'

        custom_grad = custom_tensor.grad.reshape(custom_tensor.shape).astype(np.float32)
        torch_grad = torch_tensor.grad.numpy()
        np.testing.assert_allclose(
            custom_grad, torch_grad, rtol=1e-5, atol=1e-5, err_msg='Gradient mismatch'
        )


@pytest.mark.parametrize(
    ('shape', 'dtype'),
    [
        ((2, 2), np.float32),
        ((3, 1), np.float32),
        ((5,), np.float32),  # Test 1D case
        ((1, 4, 2), np.float32),  # Test 3D case
    ],
)
def test_tensor_add(shape, dtype):
    """Tests Tensor addition (__add__) and backward pass against torch.Tensor."""
    np_data1 = np.random.randn(*shape).astype(dtype)
    np_data2 = np.random.randn(*shape).astype(dtype)
    torch_dtype = getattr(torch, np.dtype(dtype).name)

    t1 = tensor.Tensor(np_data1, dtype=dtype, requires_grad=True, shape=shape)
    t2 = tensor.Tensor(np_data2, dtype=dtype, requires_grad=True, shape=shape)
    result_t = t1 + t2

    pt1 = torch.tensor(np_data1, dtype=torch_dtype, requires_grad=True)
    pt2 = torch.tensor(np_data2, dtype=torch_dtype, requires_grad=True)
    result_pt = pt1 + pt2

    _assert_tensor_equals(result_t, result_pt)

    result_t._grads.fill(1.0)

    result_t.backward()
    result_pt.backward(torch.ones_like(result_pt))

    _assert_tensor_equals(t1, pt1, check_grad=True)
    _assert_tensor_equals(t2, pt2, check_grad=True)


@pytest.mark.parametrize(
    ('shape', 'dtype'),
    [
        ((2, 2), np.float32),
        ((3, 1), np.float32),
        ((5,), np.float32),  # Test 1D case
        ((1, 4, 2), np.float32),  # Test 3D case
    ],
)
def test_tensor_sub(shape, dtype):
    """Tests Tensor subtraction (__sub__) and backward pass against torch.Tensor."""
    np_data1 = np.random.randn(*shape).astype(dtype)
    np_data2 = np.random.randn(*shape).astype(dtype)
    torch_dtype = getattr(torch, np.dtype(dtype).name)

    t1 = tensor.Tensor(np_data1, dtype=dtype, requires_grad=True, shape=shape)
    t2 = tensor.Tensor(np_data2, dtype=dtype, requires_grad=True, shape=shape)
    result_t = t1 - t2

    pt1 = torch.tensor(np_data1, dtype=torch_dtype, requires_grad=True)
    pt2 = torch.tensor(np_data2, dtype=torch_dtype, requires_grad=True)
    result_pt = pt1 - pt2

    _assert_tensor_equals(result_t, result_pt)

    result_t._grads.fill(1.0)

    result_t.backward()
    result_pt.backward(torch.ones_like(result_pt))

    _assert_tensor_equals(t1, pt1, check_grad=True)
    _assert_tensor_equals(t2, pt2, check_grad=True)


@pytest.mark.parametrize(
    ('shape', 'dtype', 'factor'),
    [
        ((2, 2), np.float32, 0),
        ((2, 2), np.float32, 1),
        ((2, 2), np.float32, 911),
        ((2, 2), np.float32, -911),
        ((2, 2), np.float32, 3.14159),
        ((5,), np.float32, 0),
        ((5,), np.float32, 1),
        ((5,), np.float32, 911),
        ((5,), np.float32, -911),
        ((5,), np.float32, 3.14159),
        ((1, 4, 2), np.float32, 0),
        ((1, 4, 2), np.float32, 1),
        ((1, 4, 2), np.float32, 911),
        ((1, 4, 2), np.float32, -911),
        ((1, 4, 2), np.float32, 3.14159),
    ],
)
def test_tensor_mul(shape, dtype, factor):
    """Tests Tensor multiplication (__mul__) and backward pass against torch.Tensor."""
    np_data1 = np.random.randn(*shape).astype(dtype)
    torch_dtype = getattr(torch, np.dtype(dtype).name)

    t1 = tensor.Tensor(np_data1, dtype=dtype, requires_grad=True, shape=shape)
    result_t = t1 * factor

    pt1 = torch.tensor(np_data1, dtype=torch_dtype, requires_grad=True)
    result_pt = pt1 * factor

    _assert_tensor_equals(result_t, result_pt)

    result_t._grads.fill(1.0)

    result_t.backward()
    result_pt.backward(torch.ones_like(result_pt))

    _assert_tensor_equals(t1, pt1, check_grad=True)


@pytest.mark.parametrize(
    ('shape', 'dtype', 'divisor'),
    [
        ((2, 2), np.float32, 1),
        ((2, 2), np.float32, 2),
        ((2, 2), np.float32, -2),
        ((2, 2), np.float32, 3.14159),
        ((5,), np.float32, 1),
        ((5,), np.float32, 2),
        ((5,), np.float32, -2),
        ((5,), np.float32, 3.14159),
        ((1, 4, 2), np.float32, 1),
        ((1, 4, 2), np.float32, 2),
        ((1, 4, 2), np.float32, -2),
        ((1, 4, 2), np.float32, 3.14159),
    ],
)
def test_tensor_div(shape, dtype, divisor):
    """Tests Tensor division (__truediv__) and backward pass against torch.Tensor."""
    np_data1 = np.random.randn(*shape).astype(dtype)
    torch_dtype = getattr(torch, np.dtype(dtype).name)

    t1 = tensor.Tensor(np_data1, dtype=dtype, requires_grad=True, shape=shape)
    result_t = t1 / divisor

    pt1 = torch.tensor(np_data1, dtype=torch_dtype, requires_grad=True)
    result_pt = pt1 / divisor

    _assert_tensor_equals(result_t, result_pt)

    result_t._grads.fill(1.0)

    result_t.backward()
    result_pt.backward(torch.ones_like(result_pt))

    _assert_tensor_equals(t1, pt1, check_grad=True)


def test_tensor_div_zero():
    """Tests that division by zero raises ZeroDivisionError."""
    t1 = tensor.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), np.float32, True, (2, 2))

    with pytest.raises(ZeroDivisionError):
        _ = t1 / 0


@pytest.mark.parametrize(
    ('shape', 'dtype', 'factor'),
    [
        ((2, 2), np.float32, 0),
        ((2, 2), np.float32, 1),
        ((2, 2), np.float32, 11),
        ((2, 2), np.float32, -11),
        ((2, 2), np.float32, 3.14159),
        ((5,), np.float32, 0),
        ((5,), np.float32, 1),
        ((5,), np.float32, 11),
        ((5,), np.float32, -11),
        ((5,), np.float32, 3.14159),
        ((1, 4, 2), np.float32, 0),
        ((1, 4, 2), np.float32, 1),
        ((1, 4, 2), np.float32, 11),
        ((1, 4, 2), np.float32, -11),
        ((1, 4, 2), np.float32, 3.14159),
    ],
)
def test_tensor_pow(shape, dtype, factor):
    """Tests Tensor power (__pow__) and backward pass against torch.Tensor."""
    np_data1 = np.random.randn(*shape).astype(dtype)
    torch_dtype = getattr(torch, np.dtype(dtype).name)

    t1 = tensor.Tensor(np_data1, dtype=dtype, requires_grad=True, shape=shape)

    if np.any(t1.data < 0) and not isinstance(factor, int):
        with pytest.raises(ValueError):
            result_t = t1**factor
    else:
        result_t = t1**factor

        pt1 = torch.tensor(np_data1, dtype=torch_dtype, requires_grad=True)
        result_pt = pt1**factor

        _assert_tensor_equals(result_t, result_pt)

        result_t._grads.fill(1.0)

        result_t.backward()
        result_pt.backward(torch.ones_like(result_pt))

        _assert_tensor_equals(t1, pt1, check_grad=True)


def test_tensor_matmul():
    """Tests Tensor matrix multiplication (__matmul__) and backward pass."""
    # Test 2x2 @ 2x2 with deterministic data
    np_data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_data2 = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)

    t1 = tensor.Tensor(np_data1, np.float32, True, (2, 2))
    t2 = tensor.Tensor(np_data2, np.float32, True, (2, 2))
    result_t = t1 @ t2

    pt1 = torch.tensor(np_data1, dtype=torch.float32, requires_grad=True)
    pt2 = torch.tensor(np_data2, dtype=torch.float32, requires_grad=True)
    result_pt = pt1 @ pt2

    _assert_tensor_equals(result_t, result_pt)

    result_t._grads.fill(1.0)

    result_t.backward()
    result_pt.backward(torch.ones_like(result_pt))

    _assert_tensor_equals(t1, pt1, check_grad=True)
    _assert_tensor_equals(t2, pt2, check_grad=True)


def test_tensor_matmul_shape_mismatch():
    """Tests that matrix multiplication with mismatched shapes raises ValueError."""
    t1 = tensor.Tensor(np.random.randn(2, 3), np.float32, False, (2, 3))
    t2 = tensor.Tensor(np.random.randn(4, 2), np.float32, False, (4, 2))

    with pytest.raises(ValueError):
        _ = t1 @ t2


def test_tensor_matmul_3d_not_supported():
    """Tests that 3D matrix multiplication raises NotImplementedError."""
    t1 = tensor.Tensor(np.random.randn(2, 3, 4), np.float32, False, (2, 3, 4))
    t2 = tensor.Tensor(np.random.randn(2, 4, 5), np.float32, False, (2, 4, 5))

    with pytest.raises(ValueError):
        _ = t1 @ t2


def test_tensor_tensor_multiplication_not_supported():
    """Tests that tensor-tensor multiplication raises TypeError."""
    t1 = tensor.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), np.float32, False, (2, 2))
    t2 = tensor.Tensor(np.array([[0.5, 1.0], [1.5, 2.0]]), np.float32, False, (2, 2))

    with pytest.raises(TypeError):
        _ = t1 * t2


def test_tensor_add_shape_mismatch():
    """Tests that adding tensors with mismatched shapes raises ValueError."""
    shape1 = (2, 3)
    shape2 = (3, 2)
    dtype = np.float32

    t1 = tensor.Tensor(
        np.zeros(shape1, dtype=dtype), dtype=dtype, requires_grad=False, shape=shape1
    )
    t2 = tensor.Tensor(
        np.zeros(shape2, dtype=dtype), dtype=dtype, requires_grad=False, shape=shape2
    )

    with pytest.raises(ValueError):
        _ = t1 + t2


def test_tensor_sub_shape_mismatch():
    """Tests that subtracting tensors with mismatched shapes raises ValueError."""
    shape1 = (2, 3)
    shape2 = (3, 2)
    dtype = np.float32

    t1 = tensor.Tensor(
        np.zeros(shape1, dtype=dtype), dtype=dtype, requires_grad=False, shape=shape1
    )
    t2 = tensor.Tensor(
        np.zeros(shape2, dtype=dtype), dtype=dtype, requires_grad=False, shape=shape2
    )

    with pytest.raises(ValueError):
        _ = t1 - t2
