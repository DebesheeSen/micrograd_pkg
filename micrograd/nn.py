import random
from .engine import Value

class Module:
    def __init__(self):
        self.training = True 

    def train(self):
        self.training = True
        for attr in self.__dict__.values():
            if isinstance(attr,Module):
                attr.train()
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        item.train()

    def eval(self):
        self.training = False
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.eval()
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        item.eval()

    def parameters(self):
        params = []
        def collect(obj):
            if isinstance(obj, Value):
                params.append(obj)
            elif isinstance(obj, Module):
                params.extend(obj.parameters())
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        for attr in self.__dict__.values():
            collect(attr)

        return params
            
class Neuron(Module):
    def __init__(self, inputs, activation="sigmoid"):
        self.inputs = inputs
        self.w = [Value(random.uniform(-1,1)) for _ in range(self.inputs)]
        self.b = Value(0)
        assert activation in ["sigmoid","relu","tanh"]
        self.activation = activation

    def __call__(self, x): # x is not of type Value()
        out = Value(0.0)
        x_Value = [a if isinstance(a, Value) else Value(a) for a in x] # make each value of x a Value
        for i in range(self.inputs): # do w*x + b for all x
            out += self.w[i]*x_Value[i]
        out += self.b
        
        if self.activation == "sigmoid":
            return out.sigmoid()
        if self.activation == "relu":
            return out.relu()
        if self.activation == "tanh":
            return out.tanh()
        if self.activation == "leaky_relu":
            return out.leaky_relu()
        if self.activation == "softplus":  
            return out.softplus()
        if self.activation == "gelu":      
            return out.gelu()
        
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, activation):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out if len(out) > 1 else out[0]

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class SLP(Module):
    def __init__(self, nin, nouts, activation="sigmoid"):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x if isinstance(x, list) else [x])
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

class Linear(Module):
    def __init__(self, in_features, out_features, activation="sigmoid"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # weight matrix: out_features × in_features
        self.weight = [[Value(random.uniform(-1, 1)) for _ in range(in_features)] for _ in range(out_features)]

        # bias vector
        self.bias = [Value(random.uniform(-1, 1)) for _ in range(out_features)]

    def __call__(self, x):
        outputs = []
        for j in range(self.out_features):
            out = Value(0.0)
            for i in range(self.in_features):
                out += (self.weight[j][i] * x[i])
            out += self.bias[j]
            
            if self.activation == "sigmoid":
                out = out.sigmoid()
            elif self.activation == "relu":
                out = out.relu()
            elif self.activation == "tanh":
                out = out.tanh()
            elif self.activation == "leaky_relu": 
                out = out.leaky_relu()
            elif self.activation == "softplus": 
                out = out.softplus()
            elif self.activation == "gelu":     
                out = out.gelu()
        
            outputs.append(out)

        return outputs if len(outputs) > 1 else outputs[0]

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x if isinstance(x, list) else [x])
        return x


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        scale = 1.0 / (1.0 - self.p)
        out = []
        for xi in x:
            if random.random() < self.p:
                out.append(Value(0.0))
            else:
                out.append(xi * scale)
        return out

class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = [Value(1.0) for _ in range(num_features)]  # learnable scale
        self.beta  = [Value(0.0) for _ in range(num_features)]  # learnable shift
        self.running_mean = [0.0] * num_features
        self.running_var  = [1.0] * num_features

    def __call__(self, x):
        if isinstance(x[0], list):
            return self._forward_batch(x)
        else:
            return self._forward_single(x)

    def _forward_batch(self, batch):
        n = len(batch)
        out_batch = []
        for feat_idx in range(self.num_features):
            vals = [batch[i][feat_idx] for i in range(n)]
            mean = sum(v.data for v in vals) / n
            var  = sum((v.data - mean)**2 for v in vals) / n

            self.running_mean[feat_idx] = (1 - self.momentum) * self.running_mean[feat_idx] + self.momentum * mean
            self.running_var[feat_idx]  = (1 - self.momentum) * self.running_var[feat_idx]  + self.momentum * var

        result = []
        for sample in batch:
            result.append(self._normalize(sample, use_running=False))
        return result

    def _forward_single(self, x):
        return self._normalize(x, use_running=not self.training)

    def _normalize(self, x, use_running):
        out = []
        for i, xi in enumerate(x):
            mean = self.running_mean[i] if use_running else xi.data
            var  = self.running_var[i]  if use_running else 1.0
            x_hat = (xi - mean) * ((var + self.eps) ** -0.5)
            out.append(self.gamma[i] * x_hat + self.beta[i])
        return out

    def parameters(self):
        return self.gamma + self.beta