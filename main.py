from src.nn import *
from src.optim import *

def loss_fn(t, p):
    return (t - p) ** 2  # mse

X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
y = [1.0, -1.0, -1.0, 1.0]

n_input = 3
n_outs = [4, 4, 1]
mlp = MLP(n_input, n_outs)

lr = 0.01
n_epochs = 20

gd = GradientDescent(loss_fn, mlp)
mlp, loss_sch = gd.learn(n_epochs, lr, X, y)

# GRAD_DESCENT(n_epochs, lr, loss_fn, X, y, mlp)
# SGD(n_epochs, lr, loss_fn, X, y, mlp)
# MiniBatch-SGD(n_epochs, lr, loss_fn, X, y, mlp, b)
# Optimization_GD(n_epochs, lr,loss_fn, X, y, mlp, *args, **kwargs)
