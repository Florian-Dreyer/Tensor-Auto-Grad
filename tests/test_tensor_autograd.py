"""Tests for tensor autograd operations: backward, zero_grad, computation graph."""

import numpy as np
import torch

from tensor import tensor


def _assert_tensor_equals(
    custom_tensor: tensor.Tensor, torch_tensor: torch.Tensor, check_grad: bool = False
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


def test_backward_pass():
    """Tests complete backward pass through computation graph."""
    # Create a simple computation: (a + b) * c
    a = tensor.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), np.float32, True, (2, 2))
    b = tensor.Tensor(np.array([[0.5, 1.0], [1.5, 2.0]]), np.float32, True, (2, 2))
    c = 2.0

    # Compute (a + b) * c
    result = (a + b) * c

    # PyTorch equivalent
    pt_a = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True
    )
    pt_b = torch.tensor(
        [[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32, requires_grad=True
    )
    pt_result = (pt_a + pt_b) * c

    # Compare forward pass
    _assert_tensor_equals(result, pt_result)

    # Backward pass
    result.backward()
    pt_result.backward(torch.ones_like(pt_result))

    # Compare gradients
    _assert_tensor_equals(a, pt_a, check_grad=True)
    _assert_tensor_equals(b, pt_b, check_grad=True)


def test_zero_grad():
    """Tests zero_grad method."""
    t = tensor.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), np.float32, True, (2, 2))

    # Set some gradients
    t._grads = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32).flatten()

    # Check initial gradients
    assert not np.allclose(t._grads, 0.0)

    # Zero gradients
    t.zero_grad()

    # Check that gradients are zero
    assert np.allclose(t._grads, 0.0)


def test_gradient_accumulation():
    """Tests that gradients accumulate correctly across multiple backward passes."""
    a = tensor.Tensor(np.array([[1.0, 2.0]]), np.float32, True, (1, 2))
    b = tensor.Tensor(np.array([[0.5, 1.0]]), np.float32, True, (1, 2))

    # First computation
    result1 = a + b
    result1.backward()

    # Check gradients after first backward
    expected_grad_a = np.ones((1, 2))
    expected_grad_b = np.ones((1, 2))

    np.testing.assert_allclose(a._grads.reshape(a.shape), expected_grad_a)
    np.testing.assert_allclose(b._grads.reshape(b.shape), expected_grad_b)

    # Second computation (gradients should accumulate)
    result2 = a * 2
    result2.backward()

    # Check accumulated gradients
    expected_grad_a_accumulated = np.ones((1, 2)) + 2 * np.ones((1, 2))  # 1 + 2 = 3
    np.testing.assert_allclose(a._grads.reshape(a.shape), expected_grad_a_accumulated)


def test_computation_graph():
    """Tests that computation graph is built correctly."""
    a = tensor.Tensor(np.array([[1.0]]), np.float32, True, (1, 1))
    b = tensor.Tensor(np.array([[2.0]]), np.float32, True, (1, 1))

    # Build computation graph
    c = a + b
    d = c * 3
    e = d**2

    # Check that children are correctly set
    assert a in c._children
    assert b in c._children
    assert c in d._children
    assert d in e._children

    # Check operations
    assert c._op == '+'
    assert d._op == '*'
    assert e._op == '**'


def test_backward_without_gradients():
    """Tests backward pass when requires_grad=False."""
    a = tensor.Tensor(np.array([[1.0, 2.0]]), np.float32, False, (1, 2))
    b = tensor.Tensor(np.array([[0.5, 1.0]]), np.float32, True, (1, 2))

    result = a + b

    # This should not raise an error
    result.backward()

    # Only b should have gradients
    assert np.allclose(a._grads, 0.0)
    assert not np.allclose(b._grads, 0.0)


def test_complex_computation_graph():
    """Tests a more complex computation graph."""
    # Create tensors
    x = tensor.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), np.float32, True, (2, 2))
    y = tensor.Tensor(np.array([[0.5, 1.0], [1.5, 2.0]]), np.float32, True, (2, 2))

    # Complex computation: ((x + y) * 2.0) ** 2
    result = ((x + y) * 2.0) ** 2

    # PyTorch equivalent
    pt_x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True
    )
    pt_y = torch.tensor(
        [[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32, requires_grad=True
    )
    pt_result = ((pt_x + pt_y) * 2.0) ** 2

    # Compare forward pass
    _assert_tensor_equals(result, pt_result)

    # Backward pass
    result.backward()
    pt_result.backward(torch.ones_like(pt_result))

    # Compare gradients
    _assert_tensor_equals(x, pt_x, check_grad=True)
    _assert_tensor_equals(y, pt_y, check_grad=True)


def test_backward_initialization():
    """Tests that backward properly initializes gradients."""
    a = tensor.Tensor(np.array([[1.0, 2.0]]), np.float32, True, (1, 2))
    b = tensor.Tensor(np.array([[0.5, 1.0]]), np.float32, True, (1, 2))

    result = a + b

    # Gradients should be zero initially
    assert np.allclose(a._grads, 0.0)
    assert np.allclose(b._grads, 0.0)

    # After backward, gradients should be computed
    result.backward()

    assert not np.allclose(a._grads, 0.0)
    assert not np.allclose(b._grads, 0.0)

    # Check that result gradient is initialized to ones
    assert np.allclose(result._grads, 1.0)


def test_multiple_backward_calls():
    """Tests that multiple backward calls work correctly."""
    a = tensor.Tensor(np.array([[1.0]]), np.float32, True, (1, 1))
    b = tensor.Tensor(np.array([[2.0]]), np.float32, True, (1, 1))

    result = a + b

    # First backward call
    result.backward()
    grad_a_first = a._grads.copy()
    grad_b_first = b._grads.copy()

    # Second backward call (should accumulate)
    result.backward()
    grad_a_second = a._grads.copy()
    grad_b_second = b._grads.copy()

    # Gradients should accumulate
    assert np.allclose(grad_a_second, 2 * grad_a_first)
    assert np.allclose(grad_b_second, 2 * grad_b_first)
