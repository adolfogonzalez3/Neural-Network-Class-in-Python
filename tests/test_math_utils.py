'''Tests math_utils functions.'''
import pytest
import pynn.math_utils as math_utils


def test_multiply_slices():
    '''Test multiply_slices function.'''
    result = sum(math_utils.multiply_slices(range(10), range(10)))
    assert result == sum(i**2 for i in range(10))


def test_multiply_slices_raises():
    '''Test that multiply_slices function raises an assertion.'''
    with pytest.raises(AssertionError):
        math_utils.multiply_slices(range(10), range(5))
