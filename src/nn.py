from src.node import *

class Module(ABC):
    """
        Abstract class for nueral network implementations
    """

    def __init__(self, n_inputs, n_outs) -> None:
        self.parameters = None
        super().__init__()

    def __call__(self, x):
        raise NotImplementedError

class Neuron(Module):
    """
        Basic Building Block
    """

    def __init__(self, n_input: int, n_outs=1) -> None:
        """
            :param n_inputs: Input dimensions
        """

        self.w = Value(np.random.uniform(-1, 1, size=n_input))
        self.b = Value(np.random.uniform(-1, 1, size=1))
        self.parameters = [self.w, self.b]
    
    def __call__(self, x):
        """
            Forward pass
            :param x: Input vector of n_inputs dimension
        """

        act = self.w @ x + self.b
        out = act.tanh()
        return out

class Layer(Module):
    """
        Layer of neurons
        Composite of Neuron Class
    """

    def __init__(self, n_input, n_outs) -> None:
        self.neurons = [Neuron(n_input) for _ in range(n_outs)]
        self.parameters = [param for n in self.neurons for param in n.parameters]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

class MLP(Module):
    """
        Feed Forward Neural Network
    """
    
    def __init__(self, n_input, n_outs) -> None:
        ins = [n_input] + n_outs
        self.layers = [Layer(ins[i], ins[i+1]) for i in range(len(n_outs))]
        self.parameters = [param for layer in self.layers for param in layer.parameters]
    
    def __call__(self, x):
        temp = x
        for layer in self.layers:
            temp = layer(temp)
        return temp
    
    def zero_grad(self):
        """
            Initialises the gradients of the parameters before backpropagation
        """
        
        for param in self.parameters:
            k = param.grad.shape  # type: ignore
            param.grad = np.zeros(k)

# TODO: Implement using Numpy
