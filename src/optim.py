from abc import ABC, abstractmethod
import numpy as np
from src.node import *
from src.nn import *
from dataclasses import dataclass, field
from typing import Callable, Any, Dict

# @dataclass
# class Optimization(ABC):
#     n_epochs : int
#     lr : float
#     loss_fn : Callable[..., Any]
#     X : Any = field(repr=False)
#     y : Any = field(repr=False)
#     mlp : Any = field(repr=False)

#     hparams: Dict[str, Any] = field(default_factory=dict)

class Optimization(ABC):
    def __init__(self, loss_fn: Callable[..., Any], mlp: Module) -> None:
        self.loss_fn = loss_fn
        self.mlp = mlp
        self.loss_schedule = []
    
    @abstractmethod
    def learn(self, n_epochs: int, lr: float, X: Any, y: Any,
              hparams: Dict[str, Any] | None = None, **kwargs):
        self.n_epochs = n_epochs
        self.lr = lr
        self.hparams = hparams or {}
        super().__init__(**kwargs)

        return self.mlp, self.loss_schedule

class GradientDescent(Optimization):
    def learn(self, n_epochs: int, lr: float, X, y,
              hparams: Dict[str, Any] | None = None, **kwargs):
        
        super().learn(n_epochs, lr, hparams, **kwargs)

        if len(X) != len(y):
            raise Exception(f"Dataset size mismatch: {len(X)} != {len(y)}")
        
        for epoch in range(n_epochs):
            # Forward Pass
            pred = [self.mlp(x)[0] for x in X]

            # Compute Loss
            loss: Value = sum(self.loss_fn(t, p) for t, p in zip(y, pred))  # type: ignore

            self.loss_schedule.append(loss)

            # Backward Pass
            for param in self.mlp.parameters: # type: ignore
                param.grad = 0  # so that the grad of loss calculated is only for this new iteration
            loss.backward()

            print(f"{epoch} : {loss.data}")
            
            # Update Parameters
            for param in self.mlp.parameters:  # type: ignore
                param.data += -lr * param.grad
        
        return self.mlp, self.loss_schedule

class SGD(Optimization):
    def learn(self, n_epochs: int, lr: float, X, y,
              hparams: Dict[str, Any] | None = None, **kwargs):
        
        super().learn(n_epochs, lr, hparams, **kwargs)

        if len(X) != len(y):
            raise Exception(f"Dataset size mismatch: {len(X)} != {len(y)}")
        
        N = len(X)
        for epoch in range(n_epochs):
            # shuffle the dataset
            D = list(zip(X, y))
            np.random.shuffle(D)

            for i in range(N):
                x, t = X[i], y[i]
                
                # Forward Pass
                p = self.mlp(x)[0]

                # Compute Error
                loss = self.loss_fn(t, p)

                self.loss_schedule.append(loss)

                # Backward Pass
                for param in self.mlp.parameters: # type: ignore
                    param.grad = 0  # so that the grad of loss calculated is only for this new iteration
                loss.backward()

                # Update Parameters
                for param in self.mlp.parameters:  # type: ignore
                    param.data += -lr * param.grad
        
        return self.mlp, self.loss_schedule
    
class MiniBatchSGD(Optimization):
    def learn(self, n_epochs: int, lr: float, X, y,
              hparams: Dict[str, Any] | None = None, **kwargs):
        
        super().learn(n_epochs, lr, hparams, **kwargs)

        hparams = hparams or {}
        b = hparams["b"] or 4

        if len(X) != len(y):
            raise Exception(f"Dataset size mismatch: {len(X)} != {len(y)}")
        
        N = len(X)
        for epoch in range(n_epochs):
            # shuffle the dataset
            D = list(zip(X, y))
            np.random.shuffle(D)

            for t in range(1, np.ceil(N/b)):
                batch = D[(t-1)*b + 1 : np.min(t*b, N)]
                X, y = [point[0] for point in batch], [point[1] for point in batch]
                # Forward Pass
                pred = [self.mlp(x)[0] for x in X]

                # Compute Error
                loss: Value = sum(self.loss_fn(t, p) for t, p in zip(y, pred))/len(batch)  # type: ignore

                self.loss_schedule.append(loss)

                # Backward Pass
                for param in self.mlp.parameters: # type: ignore
                    param.grad = 0  # so that the grad of loss calculated is only for this new iteration
                loss.backward()

                # Update Parameters
                for param in self.mlp.parameters:  # type: ignore
                    param.data += -lr * param.grad / len(batch)

        return self.mlp, self.loss_schedule
