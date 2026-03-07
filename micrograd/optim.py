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
