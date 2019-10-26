
import math
from random import random
from itertools import chain, islice
from array import array

from pynn.math_utils import multiply_slices


class Matrix2d:

    def __init__(self, values, rows, columns):
        self.values = array('d', values)
        self.rows = rows
        self.columns = columns
        self.shape = (rows, columns)
        if len(self.values) != rows*columns:
            raise RuntimeError(f"{len(self.values)}, {rows}, {columns}")

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.values[idx]
        elif isinstance(idx, tuple):
            if len(idx) == 2 and all(isinstance(i, (int, slice)) for i in idx):
                idx_x, idx_y = idx
                if isinstance(idx_x, int):
                    rows = self.iter_rows(idx_x, idx_x+1)
                    #rows = list(self.iter_rows())[idx_x]
                    num_rows = 1
                else:
                    #rows = list(self.iter_rows())[idx_x]
                    start = idx_x.start if idx_x.start else 0
                    stop = idx_x.stop if idx_x.stop else self.rows
                    step = idx_x.step if idx_x.step else 1
                    if start is not None and start < 0:
                        start = self.rows + start
                    if stop is not None and stop < 0:
                        stop = self.rows + stop
                    rows = self.iter_rows(start, stop, step)
                    num_rows = math.ceil((min(stop, self.rows) - start) / step)
                if isinstance(idx_y, int):
                    elements = (row[idx_y] for row in rows)
                    num_cols = 1
                else:
                    start = idx_y.start if idx_y.start else 0
                    stop = idx_y.stop if idx_y.stop else self.columns
                    step = idx_y.step if idx_y.step else 1
                    if start is not None and start < 0:
                        start = self.columns + start
                    if stop is not None and stop < 0:
                        stop = self.columns + stop
                    elements = (row[idx_y] for row in rows)
                    #num_cols = len(elements[0])
                    num_cols = math.ceil(
                        (min(stop, self.columns) - start) / step
                    )
                elements = chain.from_iterable(elements)
                elements = list(elements)
                if len(elements) == 0:
                    print(idx)
                return Matrix2d(elements, num_rows, num_cols)
        raise RuntimeError(f"Not a valid method of indexing. {idx}")

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    @classmethod
    def random(cls, rows, columns):
        values = array(
            'd', [random() for _ in range(rows*columns)]
        )
        return cls(values, rows, columns)

    @classmethod
    def zeros(cls, rows, columns):
        values = array(
            'd', [0 for _ in range(rows*columns)]
        )
        return cls(values, rows, columns)

    def iter_rows(self, start=None, stop=None, step=None):
        start = start or 0
        stop = min(stop, self.rows) if stop else self.rows
        step = step or 1
        yield from (
            self.values[i*self.columns:(i+1)*self.columns]
            for i in range(start, stop, step)
        )

    def get_rows(self, start=None, stop=None, step=None):
        start = start or 0
        stop = min(stop, self.rows) if stop else self.rows
        step = step or 1
        return [
            self.values[i*self.columns:(i+1)*self.columns]
            for i in range(start, stop, step)
        ]

    def iter_cols(self, start=None, stop=None, step=None):
        start = 0
        stop = stop or self.columns
        step = step or 1
        yield from (
            self.values[i::self.columns]
            for i in range(start, stop, step)
        )

    def get_cols(self, start=None, stop=None, step=None):
        start = 0
        stop = stop or self.columns
        step = step or 1
        return [
            self.values[i::self.columns]
            for i in range(start, stop, step)
        ]

    def transpose(self):
        return Matrix2d(
            chain.from_iterable(self.iter_cols()), self.columns, self.rows
        )

    def exp(self):
        return Matrix2d((math.exp(i) for i in self), self.rows, self.columns)

    def __matmul__(self, other):
        new_values = (
            (
                sum(multiply_slices(row, other_column))
                for other_column in other.iter_cols()
            )
            for row in self.iter_rows()
        )
        new_values = array(
            'd', list(chain.from_iterable(new_values))
        )
        return Matrix2d(new_values, self.rows, other.columns)

    def __add__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i+j for i, j in zip(self, other)],
                self.rows, self.columns)
        else:
            return Matrix2d((i+other for i in self), self.rows, self.columns)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i*j for i, j in zip(self, other)],
                self.rows, self.columns)
        else:
            return Matrix2d((i*other for i in self), self.rows, self.columns)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i/j for i, j in zip(self, other)],
                self.rows, self.columns)
        else:
            return Matrix2d((i/other for i in self), self.rows, self.columns)

    def __rtruediv__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [j/i for i, j in zip(self, other)],
                self.rows, self.columns)
        else:
            return Matrix2d([other/i for i in self], self.rows, self.columns)

    def __sub__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i-j for i, j in zip(self, other)],
                 self.rows, self.columns)
        else:
            return Matrix2d([i-other for i in self], self.rows, self.columns)

    def __rsub__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i-j for i, j in zip(self, other)],
                 self.rows, self.columns)
        else:
            return Matrix2d([other-i for i in self], self.rows, self.columns)

    def __abs__(self):
        return Matrix2d([abs(i) for i in self], self.rows, self.columns)

    def __pow__(self, other):
        return Matrix2d([i**other for i in self], self.rows, self.columns)

    def __rpow__(self, other):
        return Matrix2d([other**i for i in self], self.rows, self.columns)

    def __neg__(self):
        return Matrix2d([-i for i in self], self.rows, self.columns)

    def __repr__(self):
        return f'Matrix2d({repr(self.values)})'

    def __lt__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i < j for i, j in zip(self, other)],
                 self.rows, self.columns)
        else:
            return Matrix2d([i < other for i in self], self.rows, self.columns)

    def __le__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i <= j for i, j in zip(self, other)],
                 self.rows, self.columns)
        else:
            return Matrix2d([i <= other for i in self], self.rows, self.columns)

    def __eq__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i == j for i, j in zip(self, other)],
                 self.rows, self.columns)
        else:
            return Matrix2d([other == i for i in self], self.rows, self.columns)

    def __ne__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i != j for i, j in zip(self, other)],
                 self.rows, self.columns)
        else:
            return Matrix2d([other != i for i in self], self.rows, self.columns)

    def __ge__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i >= j for i, j in zip(self, other)],
                 self.rows, self.columns)
        else:
            return Matrix2d([i >= other for i in self], self.rows, self.columns)

    def __gt__(self, other):
        if isinstance(other, Matrix2d):
            return Matrix2d(
                [i > j for i, j in zip(self, other)],
                 self.rows, self.columns)
        else:
            return Matrix2d([i > other for i in self], self.rows, self.columns)

    def apply(self, func):
        return Matrix2d([func(i) for i in self], self.rows, self.columns)

    def apply_with(self, other, func):
        return Matrix2d(
            [func(i, j) for i, j in zip(self, other)],
            self.rows, self.columns
        )
