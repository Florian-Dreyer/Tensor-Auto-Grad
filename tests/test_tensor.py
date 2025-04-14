import numpy as np
import pytest
import torch

from tensor.tensor import Tensor


def _assert_tensor_equals(
    custom_tensor: Tensor, torch_tensor: torch.Tensor, check_grad: bool = False
):
    """Compares custom Tensor data and grads (if requested) with a PyTorch tensor."""
    # data
    custom_data = custom_tensor.data.reshape(custom_tensor.shape)
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

        custom_grad = custom_tensor.grad.reshape(custom_tensor.shape)
        torch_grad = torch_tensor.grad.numpy()
        np.testing.assert_allclose(
            custom_grad, torch_grad, rtol=1e-5, atol=1e-5, err_msg='Gradient mismatch'
        )


@pytest.mark.parametrize(
    'shape, dtype',
    [
        ((2, 2), np.float32),
        ((3, 1), np.float64),
        ((5,), np.float32),  # Test 1D case
        ((1, 4, 2), np.float64),  # Test 3D case
    ],
)
def test_tensor_add(shape, dtype):
    """Tests Tensor addition (__add__) and backward pass against torch.Tensor."""
    np_data1 = np.random.randn(*shape).astype(dtype)
    np_data2 = np.random.randn(*shape).astype(dtype)
    torch_dtype = getattr(torch, np.dtype(dtype).name)

    t1 = Tensor(np_data1, dtype=dtype, requires_grad=True, shape=shape)
    t2 = Tensor(np_data2, dtype=dtype, requires_grad=True, shape=shape)
    result_t = t1 + t2

    pt1 = torch.tensor(np_data1, dtype=torch_dtype, requires_grad=True)
    pt2 = torch.tensor(np_data2, dtype=torch_dtype, requires_grad=True)
    result_pt = pt1 + pt2

    assert_tensor_equals(result_t, result_pt)

    result_t._grads.fill(1.0)

    result_t.backward()
    result_pt.backward(torch.ones_like(result_pt))

    assert_tensor_equals(t1, pt1, check_grad=True)
    assert_tensor_equals(t2, pt2, check_grad=True)
