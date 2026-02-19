import math
import random
from flask import g
import numpy as np
from abc import ABC, abstractmethod

class Operation(ABC):
    """
        Abstract Class defining the functionality for a differentiable operation
    """

    def __init__(self, label) -> None:
        self.label = label
    
    @abstractmethod
    def __call__(self, data1, data2):
        """
            :param data1 - operand 1
            :param data2 - operand 2

            Returns the computation between the operands
        """
        raise NotImplementedError
    
    @abstractmethod
    def _backward(self, grad_out):
        """
            :param grad_out - gradient of the result

            Returns a tuple of corresponding gradient changes for the operands
            leading to the result (to be used in backpropagation)
        """

        raise NotImplementedError

class OperationFactory:
    """
        Takes an Operation object and implements its functions to the computational graph nodes
    """

    def __call__(self, op, node1, node2 = None):
        """
            :param op - Operation to be carried out
            :param node1 - operand node 1
            :param node2 - operand node 2 (can be None for unary operations)
        """

        if node2 is not None:  # Binary Operation
            if not isinstance(node2, Value):
                node2 = Value(node2)
            
            def _backward():
                """
                    This function implements the backward propagation of the gradients from one node
                    to its parent nodes
                """

                grad1_update, grad2_update = op._backward(out.grad)
                node1.grad += grad1_update
                node2.grad += grad2_update
            
            out = Value(op(node1.data, node2.data), _childern = (node1, node2), _op = op.label)
            out._backward = _backward
        else:  # Unary Operation
            def _backward():
                grad1_update, _ = op._backward(out.grad)
                node1.grad += grad1_update
            
            out = Value(op(node1.data, None), _childern = (node1, ), _op = op.label)
            out._backward = _backward
        return out

class Addition(Operation):
    def __init__(self, label = "add") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        return data1 + data2
    
    def _backward(self, grad_out):
        return (grad_out, grad_out)

class Multiplication(Operation):
    def __init__(self, label = "mul") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        self.data1, self.data2 = data1, data2

        if self.data1.ndim == 0 or self.data2.ndim == 0:
            return np.array(data1 * data2)
        return np.array(data1 @ data2)

    def _backward(self, grad_out):
        d1 = np.atleast_2d(self.data1)
        d2 = np.atleast_2d(self.data2)
    
        if grad_out.ndim == 0 or grad_out.size == 1:
            grad1 = grad_out * d2
            grad2 = d1 * grad_out
        else:
            grad1 = grad_out @ d2
            grad2 = d1 @ grad_out

        return grad1.reshape(self.data1.shape), grad2.reshape(self.data2.shape)

class Subtraction(Operation):
    def __init__(self, label = "sub") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        return data1 - data2

    def _backward(self, grad_out):
        return (grad_out, grad_out)

class Division(Operation):
    def __init__(self, label = "div") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        if data2.ndim == 0 and (data2 == 0).all():
            raise ZeroDivisionError

        self.data1, self.data2 = data1, data2
        return data1 / data2
    
    def _backward(self, grad_out):
        g_out = np.atleast_2d(grad_out)
        d1 = np.atleast_2d(self.data1)
        d2 = np.atleast_2d(self.data2)

        grad1 = grad_out / self.data2

        condition = grad_out.ndim == 0 or grad_out.size == 1
        grad2 = - d1 * g_out / (d2 ** 2) if condition else - d1 @ g_out / (d2 ** 2)

        return grad1.reshape(self.data1.shape), grad2.reshape(self.data2.shape)  # CAREFUL

class Power(Operation):
    def __init__(self, label = "pow") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        if data2.ndim != 0:
            raise TypeError
        
        self.data1, self.data2 = data1, data2
        return data1 ** data2
    
    def _backward(self, grad_out):
        g_out = np.atleast_2d(grad_out)
        d1 = np.atleast_2d(self.data1)
        d2 = np.atleast_2d(self.data2)

        grad1 = g_out @ d2 * (d1 ** (d2 - 1))
        grad1 = grad1.reshape(self.data1.shape)
        return (grad1, 0.0)  # CAREFUL

class Exp(Operation):
    def __init__(self, label = "exp") -> None:
        super().__init__(label)
    
    def __call__(self, data1, _):
        self.out: np.ndarray = np.exp(data1)
        self.data = data1
        return self.out
    
    def _backward(self, grad_out):
        g_out = np.atleast_2d(grad_out)
        ou1 = np.atleast_2d(self.out)
        
        condition = grad_out.ndim == 0 or grad_out.size == 1
        grad1 = ou1 @ g_out if not condition else ou1 * grad_out
        grad1 = grad1.reshape(self.data.shape)
        return (grad1, None)  # None as unary operation  # CAREFUL

class Tanh(Operation):
    def __init__(self, label = "tanh") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        self.out: np.ndarray = np.tanh(data1)
        return self.out
    
    def _backward(self, grad_out):
        return ((1 - self.out ** 2) * grad_out, None)

class ReLU(Operation):
    def __init__(self, label = "relu") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        self.data:np.ndarray = data1
        return np.where(self.data > 0, self.data, 0.0)
    
    def _backward(self, grad_out):
        return (np.where(self.data > 0, grad_out, 0.0), None)

class LossFunction(ABC):
    """
        Define the functionality of a loss function
        No backward call as this function is considered as a composite function
    """

    def __init__(self, label) -> None:
        self.label = label

    @abstractmethod
    def __call__(self, data1, data2):
        raise NotImplementedError

class MSE(LossFunction):
    """
        Euclidean or L2 Norm
    """

    def __init__(self, label = "mse") -> None:
        super().__init__(label)

    def __call__(self, data1, data2):
        return (data1 - data2) ** 2
    
class MAE(LossFunction):
    """
        Manhattan or L1 Norm
    """

    def __init__(self, label = "mae") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        return np.where(data1 > data2, data1 - data2, data2 - data1)

class Value:
    """
        Basic node in the computational graph
    """

    def __init__(self, data, label="", _childern = (), _op = '') -> None:
        """
            Value object to store numerical values
            :param data      - numerical value
            :param label     - label for human readability
            :param _childern - all of the childern of the current value node
            :param _op       - operation leading to the current value
        """

        self.data = np.array(data, dtype=np.float64)
        print(label, self.data, type(self.data))
        self.label = label
        self._prev = set(_childern)  # used for backprop (childern is previous)
        self._op = _op
        self.op_fact = OperationFactory()  # to create operations
        self.grad = np.zeros_like(data, dtype=np.float64)  # records the partial derivative of output wrt this node
        self._backward = lambda : None  # used for backpropagation
    
    def __repr__(self) -> str:
        return f"{self.data}"

    def operate(self, other, op):
        """
            :param other - other Value object
            :param op    - defined Operation object

            Creates a node with custom operation
        """

        out = self.op_fact(op, node1=self, node2=other)
        return out

    def __add__(self, other):
        add = Addition()
        return self.operate(other, add)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        mul = Multiplication()
        return self.operate(other, mul)

    def __matmul__(self, other):
        return self * other
        
    def __rmul__(self, other):  # other * self
        return self * other    
    
    def __sub__(self, other):
        sub = Subtraction()
        return self.operate(other, sub)

    def __rsub__(self, other):
        return self - other
    
    def __truediv__(self, other):
        division = Division()
        return self.operate(other, division)

    def __pow__(self, other): # self ** other
        pow = Power()
        return self.operate(other, pow)

    def __rpow__(self, other):  # a ^ x
        other = Value(other)
        return other ** self
    
    def exp(self):
        exp = Exp()
        return self.operate(None, exp)

    def __le__(self, other):
        return self.data <= other.data

    def __gt__(self, other):
        return self.data > other.data

    def __lt__(self, other):
        return self.data < other.data
    
    def __ge__(self, other):
        return self.data >= other.data

    def act(self, func):
        """
            Activation function
        """

        out = self.op_fact(func, self, None)
        return out

    def tanh(self):
        tanh = Tanh()
        return self.act(tanh)

    def relu(self):
        relu = ReLU()
        return self.act(relu)

    def backward(self):
        """
            Performs back propagation on the current node
            To be used only on the final node (sink) of the graph
        """

        def build_topo(node):
            """
                Build a reverse topological sort of the graph
            """

            if node not in visited:
                visited.add(node)
                topo.append(node)  # WHERE IT MIGHT GO WRONG (BUT DOES IT THO?)
                for child in node._prev:
                    build_topo(child)
        
        topo = []
        visited = set()
        self.grad = np.array(1.0, dtype=np.float64)
        
        build_topo(self)
        for node in topo:
            node._backward()
            
if __name__ == "__main__":
    pass
    # v = Value(np.array([[1, 2], [2, 3]]))
    # b = Value(np.array([[3, 4], [5, 6]]))

    # a = v + b
    # print(a)
    # a.backward()
    # print(v.grad, b.grad, a.grad)

    # a = v * b
    # print(a)
    # print(v.grad, b.grad, a.grad)
    # a.backward()
    # print(v.grad, b.grad, a.grad)

    # a = v / b
    # print(a)
    # a.backward()
    # print(v.grad, b.grad, a.grad)

    # a = v - b
    # print(a)
    # a.backward()
    # print(v.grad, b.grad, a.grad)

    # a = v ** 2
    # print(a)
    # a.backward()
    # print(v.grad, b.grad, a.grad)

    # a = v.exp()
    # print(a)
    # a.backward()
    # print(v.grad, b.grad, a.grad)

    # a = v.tanh()
    # print(a)
    # a.backward()
    # print(v.grad, b.grad, a.grad)

    # a = v.relu()
    # print(a)
    # a.backward()
    # print(v.grad, b.grad, a.grad)
