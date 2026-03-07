
class Module:
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
        
            outputs.append(out)

        return outputs if len(outputs) > 1 else outputs[0]

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x if isinstance(x, list) else [x])
        return x