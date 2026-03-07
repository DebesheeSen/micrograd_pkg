import math 

class Value():
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op
        
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')
    
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
    
        out._backward = backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * (self ** -1)

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, _children=(self,), _op='^')
    
        def backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
    
        out._backward = backward
        return out

    def __neg__(self):
        out = Value(-self.data, _children=(self,), _op="neg")
    
        def backward():
            self.grad += -1 * out.grad
    
        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), _children = (self,), _op="exp")

        def backward():
            self.grad += out.data * out.grad
        out._backward = backward
        return out

    def log(self):
        out = Value(math.log(self.data), _children=(self,), _op="log")
    
        def backward():
            self.grad += (1 / self.data) * out.grad
    
        out._backward = backward
        return out
    
    def relu(self):
        out = Value(self.data if self.data > 0 else 0, _children = (self,), _op="relu")

        def backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = backward
        return out

    def sigmoid(self):
        return 1/(1+(-self).exp())

    def tanh(self):
        en = (-self).exp()
        ep = self.exp()

        return (ep - en) / (en + ep)
    
    def backward(self):
        graph = []
        visited = set()

        def build_graph(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_graph(child)

                graph.append(node)

        build_graph(self)
        self.grad = 1
        for node in reversed(graph):
            node._backward()
