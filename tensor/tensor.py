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
        self.strides = self.compute_strides(shape)
        self._shape = shape
        self._backward = lambda: None
        self._grads = (
            _grads if _grads.size > 0 else np.zeros_like(self.data, dtype=np.float32)
        )
        self._children = _children
        self._op = _op
        self._max_dim = 2

    def __repr__(
        self,
    ) -> str:
        """Function that returns a nice string representation of the tensor object."""
        return f"""Tensor(data={self.data.reshape(self.shape)},
                          dtype={self.data.dtype},
                          requires_grad={self.requires_grad},
                          shape={self._shape})"""

    @staticmethod
    def compute_strides(
        shape: tuple,
    ) -> tuple:
        """Function to compute strides for current shape.

        Args:
            shape (tuple): current shape of object.

        Returns:
            The strides for the given shape.
        """
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        return tuple(strides)

    def flatten_index(
        self,
        idx: tuple,
    ) -> int:
        """Flattens index from nd to 1d.

        Args:
            idx (tuple): nd index.

        Returns:
            Flattened index.
        """
        if len(idx) != len(self.shape):
            raise IndexError('Wrong number of indices')
        flat = 0
        for i, s, st in zip(idx, self.shape, self.strides):
            if not 0 <= i < s:
                raise IndexError('Index out of bounds')
            flat += i * st
        return flat

    def __getitem__(
        self,
        idx: tuple,
    ) -> any:
        """Implements tensor[idx] logic.

        Args:
            idx (tuple): Index to get element from.

        Returns:
            Elements at given index.
        """
        if isinstance(idx, int):
            idx = (idx,)
        return self.data[self.flatten_index(idx)]

    def __setitem__(
        self,
        idx: tuple,
        value: any,
    ) -> None:
        """Implements tensor[idx]=x logic.

        Args:
            idx (tuple): index to set element to.
            value (self.data.dtype): element to set at index.

        Returns:
            None.
        """
        if isinstance(idx, int):
            idx = (idx,)
        self.data[self.flatten_index(idx)] = value

    def __add__(
        self,
        other: Tensor,
    ) -> Tensor:
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

    def __mul__(
        self,
        other: int | float,
    ) -> Tensor:
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

    def __pow__(
        self,
        other: int | float,
    ) -> Tensor:
        """Performs power operation on the tensor with the given scalar value.

        Args:
            other (int | float): Scalar, numerical value to be used as exponent.

        Returns:
            New Tensor object with the exponentiated Tensor.

        Raises:
            ValueError if some value in data < 0 and other not int.
        """
        if np.any(self.data < 0) and not isinstance(other, int):
            raise ValueError('Cant take the root of a negative number!')

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

        if out.requires_grad:
            out._backward = _backward

        return out

    def __matmul__(
        self,
        other: Tensor,
    ) -> Tensor:
        """Performs matrix multiplication operation with self and given tensor.

        Args:
            other (Tensor): Scalar, numerical value to be used as exponent.

        Returns:
            New Tensor object with the matrix multiplication product.

        Raises:
            ValueError if shapes do not match (n x m) @ (m x k).
        """
        if self.shape[1] != other.shape[0]:
            raise ValueError('Shape mismatch!')

        if (len(self.shape) != self._max_dim) or (len(other.shape) != self._max_dim):
            # For simplicity, only 2D matrices allowed for now.
            raise NotImplementedError('Matmul currently only supports 2D tensors.')

        self_shaped = self.data.reshape(self.shape)
        other_shaped = other.data.reshape(other.shape)
        out_data = self_shaped @ other_shaped
        out_data = out_data.flatten()
        out_dtype = np.result_type(self.dtype, type(other))
        out_shape = (self.shape[0], other.shape[1])
        out = Tensor(
            data=out_data,
            dtype=out_dtype,
            requires_grad=(self.requires_grad or other.requires_grad),
            shape=out_shape,
            _children={
                self,
            },
            _op='@',
        )

        def _backward():
            dO = out.grad.reshape(out_shape)
            if self.requires_grad:
                dS = dO @ other_shaped.T
                self._grads += dS.flatten()
            if other.requires_grad:
                dS = self_shaped.T @ dO
                other._grads += dS.flatten()

        if out.requires_grad:
            out._backward = _backward

        return out

    @property
    def shape(
        self,
    ) -> tuple:
        """Getter for shape."""
        return self._shape

    @shape.setter
    def shape(
        self,
        value: tuple,
    ):
        """Setter for shape, checks compatability."""
        if math.prod(value) != len(self.data):
            raise ValueError('Shape of data and new shape and not compatible!')
        self._shape = value

    @property
    def dtype(
        self,
    ) -> np.dtype:
        """Getter for dtype."""
        return self.data.dtype

    @property
    def grad(
        self,
    ) -> np.ndarray:
        """Getter for grad."""
        return self._grads

    def view(
        self,
        new_shape: tuple,
    ) -> Tensor:
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

    def backward(
        self,
    ):
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

    def zero_grad(
        self,
    ):
        """Function to set the tensors grad to zero."""
        self._grads = np.zeros_like(self._grads)

    def _set_data(
        self,
        data: np.ndarray,
        dtype: np.dtype,
        shape: tuple,
        grads: np.ndarray,
    ) -> np.ndarray:
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
        # If grad is already set no need to flatten data
        if grads.size == 0 or grads is None:
            data = data.flatten().astype(dtype)
        if number_total_elements != len(data):
            raise ValueError('Shape of data and shape and not compatible!')

        return data
