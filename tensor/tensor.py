from __future__ import annotations

import math

import numpy as np


class Tensor:
    """A simple tensor class that supports basic operations."""

    def __init__(
        self,
        data: np.ndarray,
        dtype: np.dtype,
        requires_grad: bool,
        shape: tuple,
        _grads: np.ndarray = np.array([]),
        _children: set = set(),
        _op: str = '',
    ):
        self.data = self._set_data(data, dtype, shape, _grads)
        self.requires_grad = requires_grad
        if math.prod(shape) != len(self.data):
            raise ValueError('Shape of data and new shape and not compatible!')
        self._shape = shape
        self._backward = lambda: None
        self._grads = (
            _grads if _grads.shape[0] > 0 else np.ones_like(self.data, dtype=dtype)
        )
        self._children = _children
        self._op = _op

    def __repr__(self):
        """Function that returns a nice string representation of the tensor object."""
        return f"""Tensor(data={self.data.reshape(self.shape)},
                          dtype={self.data.dtype},
                          requires_grad={self.requires_grad},
                          shape={self._shape})"""

    def __add__(self, other: Tensor):
        """Performs elementwise addtion for the two given tensors.

        Args:
            other (Tensor): Other Tensor to be added to self, needs to have same shape.

        Returns:
            New Tensor object with the sum of self and other.

        Raises:
            ValueError if the shape of other does not match the tensors shape
        """
        if self.shape != other.shape:
            raise ValueError('Shapes do not match!')

        out_data = self.data + other.data
        out_dtype = np.result_type(self.dtype, other.dtype)
        out = Tensor(
            data=out_data,
            dtype=out_dtype,
            requires_grad=(self.requires_grad or other.requires_grad),
            shape=self.shape,
            _children={self, other},
            _op='+',
        )

        def _backward():
            """Function to perform backward step on tensors.
            Assumes out has already computed grad.
            """
            if self.requires_grad:
                self._grads += out._grads
            if other.requires_grad:
                other._grads += out._grads

        if out.requires_grad:
            out._backward = _backward

        return out

    def __mul__(self, other: int | float):
        """Performs elementwise multiplication on the tensor with the given scalar.

        Args:
            other (int | float): Scalar, numerical value to be used
                                 for element-wise multiplication.

        Returns:
            New Tensor object with the multiplied Tensor.
        """
        out_data = self.data * other
        out_dtype = np.result_type(self.dtype, type(other))
        out = Tensor(
            data=out_data,
            dtype=out_dtype,
            requires_grad=self.requires_grad,
            shape=self.shape,
            _children={
                self,
            },
            _op='*',
        )

        def _backward():
            """Function to perform backward step on tensors.
            Assumes out has already computed grad.
            """
            if self.requires_grad:
                self._grads += out._grads * other

        if out.requires_grad:
            out._backward = _backward

        return out

    def __pow__(self, other: int | float):
        """Performs power operation on the tensor with the given scalar value.

        Args:
            other (int | float): Scalar, numerical value to be used as exponent.

        Returns:
            New Tensor object with the exponentiated Tensor.
        """
        out_data = self.data**other
        out_dtype = np.result_type(self.dtype, type(other))
        out = Tensor(
            data=out_data,
            dtype=out_dtype,
            requires_grad=self.requires_grad,
            shape=self.shape,
            _children={
                self,
            },
            _op='**',
        )

        def _backward():
            self._grads += (other * self.data ** (other - 1)) * out._grads

        out._backward = _backward

        return out

    @property
    def shape(self):
        """Getter for shape."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple):
        """Setter for shape, checks compatability."""
        if math.prod(value) != len(self.data):
            raise ValueError('Shape of data and new shape and not compatible!')
        self._shape = value

    @property
    def dtype(self):
        """Getter for dtype."""
        return self.data.dtype

    @property
    def grad(self):
        """Getter for grad."""
        return self._grads

    def view(self, new_shape: tuple):
        """Function to return a different view on the same data.

        Args:
            new_shape (tuple): The shape for the view.

        Returns:
            New Tensor object with new_shape as shape but
            referenceto the same data and grads.

        Raises:
            ValueError if new_shape is not compatible to the data.
        """
        if math.prod(new_shape) != math.prod(self.shape):
            raise ValueError('Shape of data and new_shape and not compatible!')
        return Tensor(
            data=self.data,
            requires_grad=self.requires_grad,
            shape=new_shape,
            dtype=self.dtype,
            _grads=self._grads,
        )

    def backward(self):
        """Function to perform the backward pass."""
        # Sort performed operations in topological order
        operations = []
        visited_operations = set()

        def sort_topological(operation):
            if operation not in visited_operations:
                visited_operations.add(operation)
                for child in operation._children:
                    sort_topological(child)
                operations.append(operation)

        sort_topological(self)

        # Go backwards through the graph and apply chainrule
        for operation in operations[::-1]:
            operation._backward()

    def zero_grad(self):
        """Function to set the tensors grad to zero."""
        self._grads = np.zeros_like(self._grads)

    def _set_data(
        self, data: np.ndarray, dtype: np.dtype, shape: tuple, grads: np.ndarray
    ):
        """Function to return a correct data ndarray and handle shape representation.

        Args:
            data (np.ndarray): Array with the data to be represented in Tensor object.
            shape (tuple): Shape to be used for the Tensor object.
            grads (np.ndarray): Indicates if Tensor object is referencing
                                to data and grads of other Tensor object.

        Returns:
            A one-dimensional numpy array containing the elements from data.
            If grads != None it just returns the input data array.

        Raises:
            ValueError if shape of data and shape and are not compatible.
        """
        data = np.array(data)
        number_total_elements = math.prod(shape)
        if not grads:
            data = data.flatten().astype(dtype)
        if number_total_elements != len(data):
            raise ValueError('Shape of data and shape and not compatible!')

        return data


print('Test')
