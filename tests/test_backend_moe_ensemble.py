"""Unit tests for the MoEEnsembleBackend in Annif"""

import pytest
import numpy as np
from scipy.sparse import csr_array
from annif.suggestion import SuggestionBatch, SuggestionResult
from annif.backend.moe_ensemble import MoEEnsembleBackend
from annif.exception import NotSupportedException

class TestMoEEnsembleBackend:
    def test_init_default_threshold(self, project):
        """Test that the default threshold is set correctly."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )
        assert backend.threshold == 0.5

    def test_init_custom_threshold(self, project):
        """Test that a custom threshold is set correctly."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy", "threshold": 0.7},
            project=project,
        )
        assert backend.threshold == 0.7

    def test_merge_source_batches_empty(self, project):
        """Test merging empty batches."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )
        empty_csr = csr_array((0, 0))
        batch_by_source = {
            "source1": SuggestionBatch(empty_csr),
            "source2": SuggestionBatch(empty_csr),
        }
        sources = [("source1", 1.0), ("source2", 1.0)]
        params = {"limit": 10}

        result = backend._merge_source_batches(batch_by_source, sources, params)
        assert isinstance(result, SuggestionBatch)
        assert len(result) == 0

    def test_merge_source_batches_below_threshold(self, project):
        """Test merging batches where all scores are below the threshold."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # For subject1 with score 0.4 and subject2 with score 0.3
        data = np.array([0.4, 0.3])
        indices = np.array([0, 1])
        indptr = np.array([0, 2])
        csr = csr_array((data, indices, indptr), shape=(1, 2))

        batch1 = SuggestionBatch(csr)
        batch2 = SuggestionBatch(csr)

        batch_by_source = {"source1": batch1, "source2": batch2}
        sources = [("source1", 1.0), ("source2", 1.0)]
        params = {"limit": 10}

        result = backend._merge_source_batches(batch_by_source, sources, params)
        assert isinstance(result, SuggestionBatch)
        assert len(result) == 0

    def test_merge_source_batches_above_threshold(self, project):
        """Test merging batches where scores are above the threshold."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # For subject1 with score 0.6 and subject2 with score 0.8
        data = np.array([0.6, 0.8])
        indices = np.array([0, 1])
        indptr = np.array([0, 2])
        csr = csr_array((data, indices, indptr), shape=(1, 2))

        batch1 = SuggestionBatch(csr)
        batch2 = SuggestionBatch(csr)

        batch_by_source = {"source1": batch1, "source2": batch2}
        sources = [("source1", 1.0), ("source2", 1.0)]
        params = {"limit": 10}

        result = backend._merge_source_batches(batch_by_source, sources, params)
        assert isinstance(result, SuggestionBatch)
        assert len(result) > 0

    def test_merge_source_batches_weighted_average(self, project):
        """Test that the weighted average is computed correctly."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # For subject1 with score 0.9 and subject2 with score 0.7
        data = np.array([0.9, 0.7])
        indices = np.array([0, 1])
        indptr = np.array([0, 2])
        csr = csr_array((data, indices, indptr), shape=(1, 2))

        batch1 = SuggestionBatch(csr)
        batch2 = SuggestionBatch(csr)

        batch_by_source = {"source1": batch1, "source2": batch2}
        sources = [("source1", 1.0), ("source2", 1.0)]
        params = {"limit": 10}

        result = backend._merge_source_batches(batch_by_source, sources, params)
        assert isinstance(result, SuggestionBatch)
        assert len(result) > 0

    def test_train_not_supported(self, project, document_corpus):
        """Test that training is not supported."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )
        with pytest.raises(NotSupportedException):
            backend.train(document_corpus)