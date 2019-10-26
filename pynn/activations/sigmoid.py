'''Module contains sigmoid function.'''
import math

from pynn.tensor import Tensor
from pynn.matrix2d import Matrix2d


class Sigmoid(Tensor):
    def __init__(self, name="Sigmoid"):
        super().__init__(name)

    def forward(self, feed_in: Matrix2d) -> Matrix2d:
        return 1 / (1 + (-feed_in).exp())

    def backward(self, feed_in: Matrix2d, gradient: Matrix2d) -> Matrix2d:
        output = self.forward(feed_in)
        return output * (1 - output) * gradient
