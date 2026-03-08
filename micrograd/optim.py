from .engine import Value

class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.1):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data = p.data - self.lr * p.grad

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr

        self.t = 0

        self.m = [0.0 for _ in self.params]  # first moment
        self.v = [0.0 for _ in self.params]  # second moment

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

class SGDMomentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [0.0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.v[i] = self.momentum * self.v[i] + p.grad
            p.data -= self.lr * self.v[i]

class RMSProp(Optimizer):
    """RMSProp — divides lr by running average of recent gradient magnitudes.
    Good for non-stationary problems and RNNs.
    
    Args:
        lr: learning rate
        alpha: decay factor for running average (default 0.99)
        eps: small constant for numerical stability
    """
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = [0.0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * g ** 2
            p.data -= self.lr * g / (self.v[i] ** 0.5 + self.eps)


class AdaGrad(Optimizer):
    """AdaGrad — adapts lr per parameter by accumulating squared gradients.
    Good for sparse data. lr shrinks over time (can be too aggressive).
    
    Args:
        lr: learning rate
        eps: small constant for numerical stability
    """
    def __init__(self, params, lr=1e-2, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.G = [0.0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad
            self.G[i] += g ** 2
            p.data -= self.lr * g / (self.G[i] ** 0.5 + self.eps)