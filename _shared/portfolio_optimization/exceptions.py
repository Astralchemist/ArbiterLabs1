"""Custom exceptions for portfolio optimization."""


class OptimizationError(Exception):
    """Raised when cvxpy optimization fails."""

    def __init__(self, *args, **kwargs):
        default_message = "Please check your objectives/constraints or use a different solver."
        super().__init__(default_message, *args, **kwargs)


class InstantiationError(Exception):
    """Raised when there are errors instantiating pypfopt objects."""
    pass
