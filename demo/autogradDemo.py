import math

class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # 加法
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    # 乘法
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # 幂运算
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Tensor(self.data ** power, (self,), f'**{power}')

        def _backward():
            if self.requires_grad:
                self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    # 反向传播
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

x = Tensor(2.0)

y = x * 3
z = y ** 2

z.backward()

print(x)
