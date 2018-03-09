"""Unit tests for backends in Annif"""

import pytest
import annif.backend


def test_get_analyzer_nonexistent():
    with pytest.raises(ValueError):
        annif.backend.get_backend_type("nonexistent")
