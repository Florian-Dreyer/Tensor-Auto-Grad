# Tensor-Auto-Grad

A simple **tensor and autograd system** in Python, inspired by frameworks like PyTorch.  
This project implements a custom `Tensor` class with support for basic tensor operations, indexing, views, and automatic differentiation.

---

## ✨ Features

- 🧮 **Tensor data structure** based on `numpy.ndarray`  
- ➕ Basic operators: `+`, `-`, `*`, `/`, `**`, `@` (matrix multiplication)  
- 🔄 **Indexing & views** (`tensor[i, j]`, `tensor.view(new_shape)`)  
- 🧩 Automatic **gradient tracking** (similar to PyTorch’s Autograd)  
- 📈 Backpropagation with `backward()`  
- 🧹 Gradient reset with `zero_grad()`  
- 🎯 Clear error handling for shape and type mismatches  

---

## 🚀 Installation

Clone the repository and install locally (including development dependencies for linting and testing):

```bash
git clone https://github.com/<your-username>/Tensor-Auto-Grad.git
cd Tensor-Auto-Grad
pip install -e .[dev]
```

---

## 📝 Examples

### 1. Creating a Tensor
```python
import numpy as np
from tensor import Tensor

a = Tensor(data=np.array([[1, 2], [3, 4]]), dtype=np.float32, requires_grad=True, shape=(2, 2))
b = Tensor(data=np.array([[5, 6], [7, 8]]), dtype=np.float32, requires_grad=True, shape=(2, 2))

print(a)
# Tensor(data=[[1. 2.]
#              [3. 4.]], dtype=float32, requires_grad=True, shape=(2, 2))
```

---

### 2. Basic Operations
```python
c = a + b
d = a * 2
e = a @ b  # Matrix multiplication
```

---

### 3. Backpropagation
```python
loss = (a * b).sum()  # Example loss function
loss.backward()

print(a.grad)  # Gradients w.r.t. a
print(b.grad)  # Gradients w.r.t. b
```

---

## 🧪 Running Tests

All tests are located in the `tests/` directory. Run them with:

```bash
pytest
```

---

## 📦 Project Status

This project is a **learning project** to understand the fundamentals of **tensor representations** and **automatic differentiation**.  
It is **not** intended as a replacement for PyTorch, but as a minimal, educational implementation.

---

## 🛠 Roadmap

- [ ] Support for more than 2D in `matmul`  
- [ ] Additional mathematical operations (sin, exp, log, …)  
- [ ] Broadcasting for tensor operations  
- [ ] GPU support (long-term)  

---

## 📄 License

MIT License – free to use, modify, and distribute.