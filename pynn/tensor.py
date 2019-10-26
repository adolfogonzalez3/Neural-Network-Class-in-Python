'''Module that contains Tensor class and its descendants.'''

from abc import abstractmethod, ABC
from pynn.matrix2d import Matrix2d


class Tensor(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def forward(self, feed_in: Matrix2d) -> Matrix2d:
        pass

    @abstractmethod
    def backward(self, feed_in: Matrix2d, prior_out: Matrix2d) -> Matrix2d:
        pass

    def update_add(self, update: Matrix2d):
        pass
