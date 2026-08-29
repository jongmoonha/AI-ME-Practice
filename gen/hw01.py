# Generator script for "HW/HW01_Linear Regression.ipynb" and its answer notebook.
# Run this file to produce both the student distribution and the answer version.
import os
import nbformat as nbf


def build(answer):
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))

    def code(source):
        cells.append(nbf.v4.new_code_cell(source.strip("\n")))

    def problem(todo, solution):
        # student version gets the TODO scaffold, answer version gets the solution
        code(solution if answer else todo)

    # ------------------------------------------------------------ title
    md(r"""
# HW 01. Linear Regression

Estimate the parameters of a linear model and a quadratic model from noisy data.
The closed-form solution is computed with NumPy; **every gradient descent solution is written in PyTorch,
using `torch.optim`**.

Loss, residual, gradient and update used throughout:

$$J(\mathbf{w}) = \tfrac{1}{2}\|\mathbf{Xw} - \mathbf{y}\|_2^2, \qquad
\mathbf{E} = \mathbf{y} - \hat{\mathbf{y}}, \qquad \frac{\partial J}{\partial \mathbf{w}} =
-\mathbf{X}^T\mathbf{E}, \qquad \mathbf{w} \leftarrow \mathbf{w} - \rho\,\frac{\partial J}{\partial
\mathbf{w}}$$

Each problem states the **variable name your answer must be stored in**. The visualization cells are already
written and use those names, so they run as soon as the problem above them is solved.
Run the cells in order.
""")

    # ------------------------------------------------------------ data
    md(r"""
# Data Generation
### Input: x1, x2 / Output: y
""")

    code(r"""
import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

### Input data (x)
N = 100
x1 = np.random.uniform(-2, 2, (N, 1))
x2 = np.random.uniform(-2, 2, (N, 1))

### Parameters (Ground truth)
w0_gt = 0.1
w1_gt = 0.1
w2_gt = 0.3
""")

    # ------------------------------------------------------------ linear model
    md(r"""
# Linear Model
## $$y = w_{0} + w_{1} x_{1} + w_{2} x_{2}$$
""")

    code(r"""
### Output data (y) - Linear Model
y_linear = w0_gt + w1_gt * x1 + w2_gt * x2 + np.random.normal(0.0, 0.1, (N, 1))

### Data Visualization (scatter)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y_linear, c='r', s=5, alpha=0.3)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Linear Model - Data')
plt.show()
""")

    # ------------------------------------------------------------ Problem 1
    md(r"""
# Problem # 1
### Find the explicit solution using the Least Square Method

Build the augmented matrix $\mathbf{X} = [\mathbf{1},\ \mathbf{x}_1,\ \mathbf{x}_2]$ and apply $\mathbf{w}^*
= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$.

**Store your answer in `w_least_squares_linear`** — `np.ndarray`, shape `(3, 1)`.
Also keep the augmented matrix as `X_linear`, since Problem 2 uses it.
Print the three estimated parameters next to the ground truth values.
""")

    problem(r"""
### TODO: build the augmented matrix and solve the normal equation
### np.hstack stacks columns, np.linalg.inv computes the inverse

# X_linear =
# w_least_squares_linear =
""", r"""
### Augmented matrix X = [1, x1, x2]
X_linear = np.hstack([np.ones((N, 1)), x1, x2])

### Normal equation: w* = (X^T X)^-1 X^T y
w_least_squares_linear = np.linalg.inv(X_linear.T @ X_linear) @ X_linear.T @ y_linear

print('=== Least Square Method (Linear) ===')
print(f'w0 = {w_least_squares_linear[0, 0]:.6f}  (ground truth: {w0_gt})')
print(f'w1 = {w_least_squares_linear[1, 0]:.6f}  (ground truth: {w1_gt})')
print(f'w2 = {w_least_squares_linear[2, 0]:.6f}  (ground truth: {w2_gt})')
""")

    # ------------------------------------------------------------ Problem 2
    md(r"""
# Problem # 2
### Find the parameters using gradient descent, implemented in PyTorch with `torch.optim`

Do not derive the gradient by hand.
Build the loss as a tensor expression, let `loss.backward()` compute it, and let `torch.optim.SGD` apply
the update.

| Setting | Value |
|:---|:---|
| Initial weights | `np.random.randn(3, 1)` |
| Learning rate | `rho = 0.00002`, passed to the optimizer as `lr=rho` |
| Iterations | `n_iter = 3000` |
| Loss | `loss = ((y_hat - y_linear_tensor) ** 2).sum() / 2` |

`loss.backward()` **accumulates** into `.grad`, so call `optimizer.zero_grad()` every iteration.

**Store your answer in `w_optimizer_linear`** — `np.ndarray`, shape `(3, 1)`, via `.detach().numpy()`.
Print the parameters next to the ground truth.
""")

    problem(r"""
### TODO: gradient descent in PyTorch with torch.optim.SGD

# X_linear_tensor = torch.tensor(X_linear, dtype=torch.float64)
# y_linear_tensor = torch.tensor(y_linear, dtype=torch.float64)
# w_tensor        = torch.tensor(np.random.randn(3, 1), dtype=torch.float64, requires_grad=True)
#
# rho    = 0.00002
# n_iter = 3000
# optimizer =
#
# for i in range(n_iter):
#     y_hat =
#     loss  =
#     ...
#
# w_optimizer_linear =
""", r"""
### Tensors
X_linear_tensor = torch.tensor(X_linear, dtype=torch.float64)
y_linear_tensor = torch.tensor(y_linear, dtype=torch.float64)
w_tensor        = torch.tensor(np.random.randn(3, 1), dtype=torch.float64, requires_grad=True)

### Hyperparameters
rho    = 0.00002
n_iter = 3000
optimizer = torch.optim.SGD([w_tensor], lr=rho)

print(f'initial w = ({w_tensor[0, 0].item():.4f}, {w_tensor[1, 0].item():.4f}, {w_tensor[2, 0].item():.4f})')
print(f'step size rho = {rho}')
print(f'n_iter = {n_iter}')
print()

### Gradient descent
for i in range(n_iter):
    y_hat = X_linear_tensor @ w_tensor                    # y_hat = Xw
    loss  = ((y_hat - y_linear_tensor) ** 2).sum() / 2    # J = (1/2)||Xw - y||^2

    loss.backward()          # autograd computes dJ/dw
    optimizer.step()         # w <- w - rho * dJ/dw
    optimizer.zero_grad()    # clear the accumulated gradient

    if i % 500 == 0 or i == n_iter - 1:
        w_now = w_tensor.detach().numpy()
        print(f'iter={i:5d}  w0={w_now[0, 0]:+.6f}  w1={w_now[1, 0]:+.6f}  w2={w_now[2, 0]:+.6f}  J={loss.item():.6f}')

w_optimizer_linear = w_tensor.detach().numpy()

print()
print('=== Gradient Descent with torch.optim (Linear) ===')
print(f'w0 = {w_optimizer_linear[0, 0]:.6f}  (ground truth: {w0_gt})')
print(f'w1 = {w_optimizer_linear[1, 0]:.6f}  (ground truth: {w1_gt})')
print(f'w2 = {w_optimizer_linear[2, 0]:.6f}  (ground truth: {w2_gt})')
""")

    # ------------------------------------------------------------ Problem 3
    md(r"""
# Problem # 3
## Prediction Visualization (Linear Model)
### Generate more x data and compute y_hat using your estimated parameters

This cell is already written.
It uses `w_optimizer_linear` from Problem 2, so it runs as soon as that problem is solved.
The predicted surface should be a plane.
""")

    code(r"""
### Prediction inputs (many more than the training data)
N_pred  = 2000
x1_pred = np.random.uniform(-2, 2, (N_pred, 1))
x2_pred = np.random.uniform(-2, 2, (N_pred, 1))

### Compute y_hat using the parameters estimated in Problem 2
X_pred = np.hstack([np.ones((N_pred, 1)), x1_pred, x2_pred])
y_hat  = X_pred @ w_optimizer_linear

### Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1_pred, x2_pred, y_hat, c='b', s=5, alpha=0.3)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Linear Model - Prediction (N=2000)')
plt.show()
""")

    # ------------------------------------------------------------ quadratic model
    md(r"""
# Quadratic Model
## $$y = w_{0} + w_{1} x_{1}^{2} + w_{2} x_{2}^{2}$$
""")

    code(r"""
### Output data (y) - Quadratic Model
y_quadratic = w0_gt + w1_gt * x1**2 + w2_gt * x2**2 + np.random.normal(0.0, 0.1, (N, 1))

### Data Visualization (scatter)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y_quadratic, c='r', s=5, alpha=0.3)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Quadratic Model - Data')
plt.show()
""")

    # ------------------------------------------------------------ Problem 4
    md(r"""
# Problem # 4
### Find the explicit solution using the Least Square Method

Same normal equation as Problem 1.
Only the two feature columns change, to $x_1^2$ and $x_2^2$.

**Store your answer in `w_least_squares_quadratic`** — `np.ndarray`, shape `(3, 1)`.
Also keep the augmented matrix as `X_quadratic`, since Problem 5 uses it.
""")

    problem(r"""
### TODO: build the augmented matrix with squared features and solve the normal equation

# X_quadratic =
# w_least_squares_quadratic =
""", r"""
### Augmented matrix X = [1, x1^2, x2^2]
X_quadratic = np.hstack([np.ones((N, 1)), x1**2, x2**2])

### Normal equation: w* = (X^T X)^-1 X^T y
w_least_squares_quadratic = np.linalg.inv(X_quadratic.T @ X_quadratic) @ X_quadratic.T @ y_quadratic

print('=== Least Square Method (Quadratic) ===')
print(f'w0 = {w_least_squares_quadratic[0, 0]:.6f}  (ground truth: {w0_gt})')
print(f'w1 = {w_least_squares_quadratic[1, 0]:.6f}  (ground truth: {w1_gt})')
print(f'w2 = {w_least_squares_quadratic[2, 0]:.6f}  (ground truth: {w2_gt})')
""")

    # ------------------------------------------------------------ Problem 5
    md(r"""
# Problem # 5
### Find the parameters using gradient descent, implemented in PyTorch with `torch.optim`

The same loop as Problem 2, applied to the quadratic design matrix.

| Setting | Value |
|:---|:---|
| Initial weights | `np.random.randn(3, 1)` |
| Learning rate | `rho = 0.0002` |
| Iterations | `n_iter = 3000` |

The learning rate is ten times larger than in Problem 2.
Squaring the features changes the scale of $\mathbf{X}^T\mathbf{X}$, and the rate used in Problem 2 would
still be far from the solution after 3000 iterations.

**Store your answer in `w_optimizer_quadratic`** — `np.ndarray`, shape `(3, 1)`.
Print it next to the Problem 4 closed-form solution; the two should agree closely.
""")

    problem(r"""
### TODO: the same optimizer loop as Problem 2, applied to the quadratic model

# X_quadratic_tensor =
# y_quadratic_tensor =
# w_tensor           =
#
# rho    = 0.0002
# n_iter = 3000
# optimizer =
#
# for i in range(n_iter):
#     ...
#
# w_optimizer_quadratic =
""", r"""
### Tensors
X_quadratic_tensor = torch.tensor(X_quadratic, dtype=torch.float64)
y_quadratic_tensor = torch.tensor(y_quadratic, dtype=torch.float64)
w_tensor           = torch.tensor(np.random.randn(3, 1), dtype=torch.float64, requires_grad=True)

### Hyperparameters
rho    = 0.0002
n_iter = 3000
optimizer = torch.optim.SGD([w_tensor], lr=rho)

### Gradient descent
for i in range(n_iter):
    y_hat = X_quadratic_tensor @ w_tensor
    loss  = ((y_hat - y_quadratic_tensor) ** 2).sum() / 2

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 500 == 0 or i == n_iter - 1:
        w_now = w_tensor.detach().numpy()
        print(f'iter={i:5d}  w0={w_now[0, 0]:+.6f}  w1={w_now[1, 0]:+.6f}  w2={w_now[2, 0]:+.6f}  J={loss.item():.6f}')

w_optimizer_quadratic = w_tensor.detach().numpy()

print()
print('=== Gradient Descent with torch.optim (Quadratic) ===')
print(f'w0 = {w_optimizer_quadratic[0, 0]:.6f}  (least squares: {w_least_squares_quadratic[0, 0]:.6f})')
print(f'w1 = {w_optimizer_quadratic[1, 0]:.6f}  (least squares: {w_least_squares_quadratic[1, 0]:.6f})')
print(f'w2 = {w_optimizer_quadratic[2, 0]:.6f}  (least squares: {w_least_squares_quadratic[2, 0]:.6f})')
""")

    # ------------------------------------------------------------ Problem 6
    md(r"""
# Problem # 6
## Prediction Visualization (Quadratic Model)
### Generate more x data and compute y_hat using your estimated parameters

This cell is already written.
It uses `w_optimizer_quadratic` from Problem 5.
The predicted surface should now be curved.
""")

    code(r"""
### Prediction inputs (many more than the training data)
N_pred  = 2000
x1_pred = np.random.uniform(-2, 2, (N_pred, 1))
x2_pred = np.random.uniform(-2, 2, (N_pred, 1))

### Compute y_hat using the parameters estimated in Problem 5
X_pred = np.hstack([np.ones((N_pred, 1)), x1_pred**2, x2_pred**2])
y_hat  = X_pred @ w_optimizer_quadratic

### Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1_pred, x2_pred, y_hat, c='b', s=5, alpha=0.3)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Quadratic Model - Prediction (N=2000)')
plt.show()
""")

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    nb.metadata['language_info'] = {'name': 'python'}
    return nb


os.makedirs('HW/Answer', exist_ok=True)

for answer, path in [(False, 'HW/HW01_Linear Regression.ipynb'),
                     (True,  'HW/Answer/HW01_Linear Regression_answer.ipynb')]:
    nb = build(answer)
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Generated {len(nb["cells"]):2d} cells -> {path}')
