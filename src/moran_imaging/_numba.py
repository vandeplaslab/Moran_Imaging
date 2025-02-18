"""Numba utilities."""

try:
    from numba import boolean, jit, njit, prange
except (ImportError, ModuleNotFoundError):

    def jit(*dec_args, **dec_kwargs):
        """Decorator mimicking numba.jit."""

        def intercepted_function(f, *f_args, **f_kwargs):
            return f

        return intercepted_function

    njit = jit

    prange = range
    boolean = bool
