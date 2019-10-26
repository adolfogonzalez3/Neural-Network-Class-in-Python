'''Test the Matrix2d class.'''
import pytest
from random import random
from itertools import product, chain
from pynn.matrix2d import Matrix2d


@pytest.mark.parametrize(("rows", "cols"), product([5, 10, 15], repeat=2))
def test_matrix2d_iter_rows(rows, cols):
    matrix = Matrix2d(list(range(cols))*rows, rows, cols)
    print(matrix.values)
    for i, row in enumerate(matrix.iter_rows()):
        print(row)
        for j in range(cols):
            assert j == row[j]


@pytest.mark.parametrize(("rows", "cols"), product([5, 10, 15], repeat=2))
def test_matrix2d_iter_columns(rows, cols):
    matrix = Matrix2d(list(range(cols))*rows, rows, cols)
    for i, col in enumerate(matrix.iter_cols()):
        for j in range(rows):
            assert i == col[j]


@pytest.mark.parametrize(("rows", "cols"), product([5, 10, 15], repeat=2))
def test_matrix2d_random_shape(rows, cols):
    matrix = Matrix2d.random(rows, cols)
    assert matrix.shape == (rows, cols)


@pytest.mark.parametrize(("rows", "cols"), product([5, 10, 15], repeat=2))
def test_matrix2d(rows, cols):
    matrix_1 = Matrix2d.random(rows, cols)
    matrix_2 = Matrix2d.random(cols, rows)

    matrix_3 = matrix_1 @ matrix_2

    assert matrix_3.shape == (matrix_1.rows, matrix_2.columns)
    assert len(matrix_3) == (matrix_1.rows * matrix_2.columns)


def test_matrix2d_indexing():
    rows = 10
    cols = 10
    matrix = Matrix2d(list(range(cols))*rows, rows, cols)

    matrix_out = matrix[:10]
    for i, out in enumerate(matrix_out):
        assert i == out

    for i, out in enumerate(list(range(cols))*rows):
        assert matrix[i] == out

    matrix_out = matrix[:3, 1:-1]

    assert matrix_out.rows == 3
    assert matrix_out.columns == 8
    print(matrix_out)
    for i, out in enumerate(list(range(1, cols-1))*3):
        assert matrix_out[i] == out


def test_matrix2d_matmul():
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2, 3],
         [4, 5, 6]]
    BT = [[1, 4],
           [2, 5],
           [3, 6]]
    C = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    matrix_A = Matrix2d(chain.from_iterable(A), 2, 2)
    matrix_B = Matrix2d(chain.from_iterable(B), 2, 3)
    matrix_BT = Matrix2d(chain.from_iterable(BT), 3, 2)
    matrix_C = Matrix2d(chain.from_iterable(C), 3, 3)

    A_B = [[9, 12, 15],
           [19, 26, 33]]
    B_C = [[30, 36, 42],
           [66, 81, 96]]
    C_BT = [[14, ]]

    matrix_A_B = matrix_A @ matrix_B
    matrix_B_C = matrix_B @ matrix_C

    assert all(i == j for i, j in zip(matrix_A_B, chain.from_iterable(A_B)))
    assert all(i == j for i, j in zip(matrix_B_C, chain.from_iterable(B_C)))

def test_matrix2d_add():
    matrix = Matrix2d(list(range(100)), 10, 10)
    matrix = matrix + 1
    assert all((i+1) == element for i, element in enumerate(matrix))
    matrix = 1 + matrix
    assert all((i+2) == element for i, element in enumerate(matrix))


def test_matrix2d_sub():
    matrix = Matrix2d(list(range(100)), 10, 10)
    matrix = matrix - 1
    assert all((i-1) == element for i, element in enumerate(matrix))
    matrix = Matrix2d(list(range(100)), 10, 10)
    matrix = 1 - matrix
    assert all((1-i) == element for i, element in enumerate(matrix))


def test_matrix2d_mul():
    matrix = Matrix2d(list(range(100)), 10, 10)
    matrix = matrix * 10
    assert all(i*10 == element for i, element in enumerate(matrix))
    matrix = 10 * matrix
    assert all(i*100 == element for i, element in enumerate(matrix))


def test_matrix2d_abs():
    matrix = Matrix2d([-i for i in range(100)], 10, 10)
    matrix = abs(matrix)
    assert all(element >= 0 for element in matrix)


def test_matrix2d_pow():
    matrix = Matrix2d(list(range(100)), 10, 10)
    matrix = matrix**2
    assert all(i**2 == element for i, element in enumerate(matrix))

def test_matrix2d_rpow():
    matrix = Matrix2d(list(range(100)), 10, 10)
    matrix = 2**matrix
    assert all(2**i == element for i, element in enumerate(matrix))

def test_matrix2d_neg():
    matrix = Matrix2d(list(range(100)), 10, 10)
    matrix = -matrix
    assert all(-i == element for i, element in enumerate(matrix))

def test_matrix2d_cmp():
    matrix = Matrix2d(list(range(100)), 10, 10)
    assert all(matrix == matrix)
    assert all(matrix != -(matrix+1))
    assert all((matrix+1) >= -matrix)
    assert all(matrix > -(1+matrix))
    assert all(-matrix <= matrix)
    assert all(-matrix < (matrix+1))