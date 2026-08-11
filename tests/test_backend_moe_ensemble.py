"""Unit tests for the MoEEnsembleBackend in Annif"""

import numpy as np
import pytest
from scipy.sparse import csr_array

from annif.backend.moe_ensemble import MoEEnsembleBackend
from annif.exception import NotSupportedException
from annif.suggestion import SuggestionBatch


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

        csr = csr_array((data, np.array([0, 1]), np.array([0, 2])), shape=(1, 2))
        batch_by_source = {
            "source1": SuggestionBatch(csr),
            "source2": SuggestionBatch(csr),
        }
        result = backend._merge_source_batches(
            batch_by_source, [("source1", 1.0), ("source2", 1.0)], {"limit": 10}
        )

        assert isinstance(result, SuggestionBatch)
        assert len(result) == 1  # One document in the batch
        assert result.array.shape == (
            1,
            len(project.subjects),
        )  # Correct shape: (n_docs, n_subjects)

    def test_merge_source_batches_above_threshold(self, project):
        """Test merging batches where scores are above the threshold."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # For subject1 with score 0.6 and subject2 with score 0.8
        data = np.array([0.6, 0.8])

        csr = csr_array((data, np.array([0, 1]), np.array([0, 2])), shape=(1, 2))
        batch_by_source = {
            "source1": SuggestionBatch(csr),
            "source2": SuggestionBatch(csr),
        }
        result = backend._merge_source_batches(
            batch_by_source, [("source1", 1.0), ("source2", 1.0)], {"limit": 10}
        )

        assert isinstance(result, SuggestionBatch)
        assert len(result) > 0

        # Extract the first SuggestionResult and its top suggestion
        first_result = next(iter(result))
        top_suggestion = next(iter(first_result))
        assert top_suggestion.score == pytest.approx(0.8)  # Highest score in the input

    def test_merge_source_batches_weighted_average(self, project):
        """Test that the weighted average is computed correctly."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # Create a csr_array for batch1 with subject1 (score 0.9)
        data1 = np.array([0.9])
        indices1 = np.array([0])
        indptr1 = np.array([0, 1])
        csr1 = csr_array((data1, indices1, indptr1), shape=(1, 1))

        # Create a csr_array for batch2 with subject2 (score 0.6)
        data2 = np.array([0.6])
        indices2 = np.array([0])
        indptr2 = np.array([0, 1])
        csr2 = csr_array((data2, indices2, indptr2), shape=(1, 1))

        batch_by_source = {
            "source1": SuggestionBatch(csr1),
            "source2": SuggestionBatch(csr2),
        }
        result = backend._merge_source_batches(
            batch_by_source, [("source1", 1.0), ("source2", 1.0)], {"limit": 10}
        )

        assert isinstance(result, SuggestionBatch)
        assert len(result) > 0

        # Extract the first SuggestionResult and its top suggestion
        first_result = next(iter(result))
        top_suggestion = next(iter(first_result))

        # Weighted average: (0.9 * (0.9 / 1.5)) + (0.6 * (0.6 / 1.5)) = 0.78
        assert top_suggestion.score == pytest.approx(0.78)

    def test_train_not_supported(self, project, document_corpus):
        """Test that training is not supported."""
        backend = MoEEnsembleBackend(
            backend_id="moe_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )
        with pytest.raises(NotSupportedException):
            backend.train(document_corpus)
