'''Module which contains functions for performing math operations.'''


def multiply_slices(array_a, array_b):
    '''Multiply iterables elementwise.'''
    assert len(array_a) == len(array_b)
    return (a*b for a, b in zip(array_a, array_b))
