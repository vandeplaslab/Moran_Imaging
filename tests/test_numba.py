from moran_imaging._numba import njit, prange, boolean


def test_njit_decorator():
    @njit
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_prange_function():
    @njit
    def sum_range(n):
        total = 0
        for i in prange(n):
            total += i
        return total

    assert sum_range(5) == 10  # 0 + 1 + 2 + 3 + 4
    assert sum_range(10) == 45  # 0 + 1 + ... + 9

def test_boolean_type():
    @njit
    def is_even(n: int) -> boolean:
        return n % 2 == 0

    assert is_even(4) is True
    assert is_even(5) is False