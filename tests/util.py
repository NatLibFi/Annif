"""utility functions used by unit tests"""

import os
from contextlib import contextmanager


@contextmanager
def umask_context(mask):
    """Context manager to temporarily set umask"""
    original_umask = os.umask(0o777)  # Get current umask
    os.umask(mask)  # Set new umask
    try:
        yield
    finally:
        os.umask(original_umask)  # Restore original umask
