"""
Acknowledge
- Following optimizer are implemented base on paper 
  "An overview of gradient descent optimization algorithms"
"""


import numpy as np


class SGD:
    def __init__(self, lr):
        self.lr = lr
        
    def forward(self, ws, grad_ws):
        ret = []
        for i in range(len(ws)):
            ret.append(ws[i] - self.lr * grad_ws[i])
        return ret


class Momentum:
    """based on paper 
    "An overview of gradient descent optimization algorithms"
    """
    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None
        
    def forward(self, ws, grad_ws):
        ret = []
        if self.v is None:
            self.v = [0] * len(ws)
            
        for i in range(len(ws)):
            self.v[i] = self.beta * self.v[i] + self.lr * grad_ws[i]
            ret.append(ws[i] - self.v[i])
        return ret


class NestrovSGD:
    def __init__(self) -> None:
        pass
    

class Adam:
    """based on paper 
    "An overview of gradient descent optimization algorithms"
    """
    def __init__(self, lr, beta1=0.5, beta2=0.999, eps=1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None
        self.t = 1
    
    def forward(self, ws, grad_ws):
        ret = []
        if self.v is None:
            self.v = grad_ws
        if self.m is None:
            self.m = [grad_ws[i]**2 for i in range(len(grad_ws))]
        
        for i in range(len(ws)):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * grad_ws[i]
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * grad_ws[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            ret.append(ws[i] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        
        self.t += 1
        return ret