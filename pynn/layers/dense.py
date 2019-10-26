'''Module that contains Dense layer.'''
from pynn.tensor import Tensor
from pynn.matrix2d import Matrix2d

class Dense(Tensor):
    def __init__(self, feed_in: int, feed_out: int, name="Dense"):
        self.value = Matrix2d.random(feed_in, feed_out) * 1e-3 - 5e-3
        super().__init__(name)

    def forward(self, feed_in: Matrix2d) -> Matrix2d:
        return feed_in @ self.value

    def backward(self, feed_in: Matrix2d, gradient: Matrix2d) -> Matrix2d:
        return -feed_in.transpose() @ gradient

    def update_add(self, update: Matrix2d):
        self.value = self.value + update