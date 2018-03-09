"""Unit tests for analyzers in Annif"""

import pytest
import annif.analyzer

def test_get_analyzer_nonexistent():
    with pytest.raises(ValueError):
        annif.analyzer.get_analyzer("nonexistent")
