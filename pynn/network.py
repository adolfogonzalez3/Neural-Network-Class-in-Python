
from pynn.matrix2d import Matrix2d
from pynn.tensor import Tensor

class Network:
    def __init__(self):
        self.tensors = []
        self.forwards = []

    def append(self, tensor: Tensor):
        self.tensors.append(tensor)

    def forward(self, feed_in: Matrix2d) -> Matrix2d:
        self.forwards = [feed_in]
        for tensor in self.tensors:
            self.forwards.append(tensor.forward(self.forwards[-1]))
        return self.forwards.pop()

    def gradients(self, gradient: Matrix2d) -> [Matrix2d]:
        gradients = [gradient]
        iters = list(zip(self.tensors, self.forwards))
        for (tensor, forward) in reversed(iters):
            gradients.append(tensor.backward(forward, gradients[-1]))
        gradients.reverse()
        return gradients[:-1]

    def update_add(self, updates: [Matrix2d]):
        for (tensor, update) in zip(self.tensors, updates):
            tensor.update_add(update)