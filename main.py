from src.nn import *
from src.optim import *

def train_mlp():
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

    opt = GradientDescent(loss_fn, mlp)
    # opt = SGD(loss_fn, mlp)
    # opt = MiniBatchSGD(loss_fn, mlp)
    mlp, loss_sch = opt.learn(n_epochs, lr, X, y, hparams={"b": 2})

    print(loss_sch)

def train_neuron():
    x = np.array([1, 2, 3, 4])
    y = 1

    n_input = 4
    n_outs = 1
    neuron = Neuron(n_input, n_outs)

    def loss_fn(t, p):
        return (t - p) ** 2  # mse

    p = neuron(x)
    print(type(p))
    print(neuron.w.label, neuron.b.label, p.label)
    loss = loss_fn(y, p)
    print(loss.data)
    
    print(neuron.w.grad, neuron.b.grad)
    loss.backward()

if __name__ == "__main__":
    train_neuron()
