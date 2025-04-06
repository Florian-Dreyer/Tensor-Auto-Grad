import numpy as np
import math

class Tensor:
    '''A simple tensor class that supports basic operations.
    product of shape must match length of data
    '''

    def __init__(self, data: np.ndarray, dtype: np.dtype, requires_grad: bool, shape: tuple, _grads: np.ndarray = None, _children: set = (), _op: str = ''):
        self.data = self.__set_data(data, dtype, shape, _grads)
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.shape = shape
        self._backward = lambda: None
        self._grads = _grads if _grads else np.zeros_like(self.data)
        self._children = _children
        self._op = _op

    def __repr__(self):
        '''Function that returns a nice string representation of the tensor object.'''
        return f"Tensor(data={self.data.view(self.shape)}, dtype={self.dtype}, requires_grad={self.requires_grad}, shape={self.shape}"

    def __add__(self, other: Tensor):
        '''Performs elementwise addtion for the two given tensors.

        Args:
            TODO

        Returns:
            TODO

        Raises:
            ValueError if the shape of other does not match the tensors shape
        '''
        if self.shape != other.shape:
            raise ValueError('Shapes do not match!')

        out_data = self.data + other.data
        out = Tensor(out_data,
                    requires_grad=(self.requires_grad or other.requires_grad),
                    _children=(self, other),
                    _op='+')
        
        def _backward():
            '''Function to perform backward step on tensors, assumes out has already computed grad.'''
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        if out.requires_grad:
            out._backward = _backward

        return out

    def __mul__(self, other: int | float):
        '''Performs elementwise multiplication on the tensor with the given scalar value.

        Args:
            TODO

        Returns:
            TODO
        
        Raises:
            TODO
        '''
        out_data = self.data * other
        out = Tensor(out_data,
                    requires_grad=self.requires_grad,
                    _children=(self, ),
                    _op='*')
        
        def _backward():
            '''Function to perform backward step on tensors, assumes out has already computed grad.'''
            if self.requires_grad:
                self.grad += out._grads * other

        if out.requires_grad:
            out._backward = _backward

        return out
    
    @property
    def shape(self):
        '''Getter for shape.'''
        return self.shape
    
    @shape.setter
    def shape(self, value: tuple):
        '''Setter for shape, checks compatability.'''
        if math.product(value) != math.product(self.shape):
            raise ValueError('Shape of data and new shape and not compatible!')
        self.shape = value

    @property
    def dtype():
        '''Getter for dtype.'''
        return self.dtype

    def view(self, new_shape):
        '''Function to return a different view on the same data.
        
        Args:
            TODO

        Returns:
            TODO
        
        Raises:
            TODO
        '''
        if math.product(new_shape) != math.product(self.shape):
            raise ValueError('Shape of data and new_shape and not compatible!')
        return Tensor(data=self.data, requires_grad=self.requires_grad, shape=new_shape, _grads=self._grads)

    def backward(self):
        '''Function to perform the backward pass.'''

    def zero_grad(self):
        '''Function to set the tensors grad to zero.'''
        self._grads = np.zeros_like(self._grads)

    def _set_data(data: np.ndarray, shape: tuple, grads: bool):
        '''Function to return a correct data ndarray and handle shape representation.

        Args:
            TODO

        Returns:
            TODO
        
        Raises:
            TODO
        '''
        number_total_elements = math.prod(shape)
        if not grads:
            data = data.flatten().astype(dtype)
        if number_total_elements != len(data):
            raise ValueError('Shape of data and shape and not compatible!')

        return data
