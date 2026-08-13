"""Unit tests for the ThresholdEnsembleBackend in Annif"""

import numpy as np
import pytest
from scipy.sparse import csr_array

from annif.backend.threshold_ensemble import ThresholdEnsembleBackend
from annif.exception import NotSupportedException
from annif.suggestion import SuggestionBatch


class TestThresholdEnsembleBackend:
    def test_init_default_threshold(self, project):
        """Test that the default threshold is set correctly."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )
        assert backend.threshold == 0.1

    def test_init_custom_threshold(self, project):
        """Test that a custom threshold is set correctly."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.7},
            project=project,
        )
        assert backend.threshold == 0.7

    def test_merge_source_batches_empty(self, project):
        """Test merging empty batches."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
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
        """Test that documents with no activated sources produce no suggestions."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # Neither source has a prediction >= 0.5, so neither source
        # is activated for the document.
        data = np.array([0.4, 0.3])

        csr = csr_array(
            (data, np.array([0, 1]), np.array([0, 2])),
            shape=(1, 2),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr),
            "source2": SuggestionBatch(csr),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0), ("source2", 1.0)],
            {"limit": 10},
        )

        assert isinstance(result, SuggestionBatch)
        assert len(result) == 1
        assert result.array.shape == (1, 2)
        assert result.array.nnz == 0

    def test_merge_source_batches_above_threshold(self, project):
        """Test merging batches where sources are activated."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # Both sources are activated because they have predictions
        # above the threshold.
        data = np.array([0.6, 0.8])

        csr = csr_array(
            (data, np.array([0, 1]), np.array([0, 2])),
            shape=(1, 2),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr),
            "source2": SuggestionBatch(csr),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0), ("source2", 1.0)],
            {"limit": 10},
        )

        assert isinstance(result, SuggestionBatch)
        assert len(result) > 0

        first_result = next(iter(result))
        top_suggestion = next(iter(first_result))

        assert top_suggestion.score == pytest.approx(0.8)

    def test_merge_source_batches_weighted_average(self, project):
        """Test that configured source weights are used in the average."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # Both sources predict the same subject, so the result is the
        # weighted average:
        #
        # (0.9 * 2 + 0.6 * 1) / (2 + 1) = 0.8
        csr1 = csr_array(
            (np.array([0.9]), np.array([0]), np.array([0, 1])),
            shape=(1, 1),
        )
        csr2 = csr_array(
            (np.array([0.6]), np.array([0]), np.array([0, 1])),
            shape=(1, 1),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr1),
            "source2": SuggestionBatch(csr2),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 2.0), ("source2", 1.0)],
            {"limit": 10},
        )

        first_result = next(iter(result))
        top_suggestion = next(iter(first_result))

        assert top_suggestion.score == pytest.approx(0.8)

    def test_merge_source_batches_single_active_source(self, project):
        """Test that one activated source is returned unchanged."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # source1 is active; source2 is not.
        data1 = np.array([0.8, 0.2])
        data2 = np.array([0.1, 0.2])

        csr1 = csr_array(
            (data1, np.array([0, 1]), np.array([0, 2])),
            shape=(1, 2),
        )
        csr2 = csr_array(
            (data2, np.array([0, 1]), np.array([0, 2])),
            shape=(1, 2),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr1),
            "source2": SuggestionBatch(csr2),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0), ("source2", 1.0)],
            {"limit": 10},
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            csr1.toarray(),
        )

    def test_merge_source_batches_preserves_below_threshold_predictions(self, project):
        """Test that activation is thresholded but predictions are not."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # Both sources are activated because each has one prediction >= 0.5.
        #
        # The predictions below 0.5 must nevertheless be retained when
        # the original source arrays are averaged.
        csr1 = csr_array(
            (
                np.array([0.8, 0.2]),
                np.array([0, 1]),
                np.array([0, 2]),
            ),
            shape=(1, 2),
        )
        csr2 = csr_array(
            (
                np.array([0.1, 0.6]),
                np.array([0, 1]),
                np.array([0, 2]),
            ),
            shape=(1, 2),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr1),
            "source2": SuggestionBatch(csr2),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0), ("source2", 1.0)],
            {"limit": 10},
        )

        # The expected result is the average of the ORIGINAL arrays:
        #
        # subject 1: (0.8 + 0.1) / 2 = 0.45
        # subject 2: (0.2 + 0.6) / 2 = 0.40
        #
        # If filtered_batch.array were used for accumulation, the result
        # would instead be [0.4, 0.3].
        np.testing.assert_allclose(
            result.array.toarray(),
            np.array([[0.45, 0.40]]),
        )

    def test_merge_source_batches_threshold_applied_per_document(self, project):
        """Test that sources are activated independently for each document."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # source1: doc1 is below threshold, doc2 is above threshold
        csr1 = csr_array(
            (
                np.array([0.4, 0.9]),
                np.array([0, 0]),
                np.array([0, 1, 2]),
            ),
            shape=(2, 1),
        )

        # source2: doc1 is above threshold, doc2 is below threshold
        csr2 = csr_array(
            (
                np.array([0.8, 0.3]),
                np.array([0, 0]),
                np.array([0, 1, 2]),
            ),
            shape=(2, 1),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr1),
            "source2": SuggestionBatch(csr2),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0), ("source2", 1.0)],
            {"limit": 10},
        )

        assert isinstance(result, SuggestionBatch)
        assert len(result) == 2

        results = list(result)

        # Doc 1: only source2 is active.
        assert next(iter(results[0])).score == pytest.approx(0.8)

        # Doc 2: only source1 is active.
        assert next(iter(results[1])).score == pytest.approx(0.9)

    def test_merge_source_batches_recalculates_weights_per_document(self, project):
        """Test that weights are recalculated from activated sources per document."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # For doc1 both sources are activated:
        #   source1 = 0.9, weight = 2
        #   source2 = 0.6, weight = 1
        #
        # Result = (0.9 * 2 + 0.6 * 1) / 3 = 0.8
        #
        # For doc2 only source2 is activated, so its configured weight
        # is the entire denominator and its prediction is unchanged.
        csr1 = csr_array(
            (
                np.array([0.9, 0.4]),
                np.array([0, 0]),
                np.array([0, 1, 2]),
            ),
            shape=(2, 1),
        )

        csr2 = csr_array(
            (
                np.array([0.6, 0.8]),
                np.array([0, 0]),
                np.array([0, 1, 2]),
            ),
            shape=(2, 1),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr1),
            "source2": SuggestionBatch(csr2),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 2.0), ("source2", 1.0)],
            {"limit": 10},
        )

        results = list(result)

        assert next(iter(results[0])).score == pytest.approx(0.8)
        assert next(iter(results[1])).score == pytest.approx(0.8)

    def test_merge_source_batches_limit_applied_after_averaging(self, project):
        """Test that limit is applied to the final averaged predictions."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # Both sources are active. Their averaged predictions are:
        #
        # subject 0: (0.9 + 0.7) / 2 = 0.8
        # subject 1: (0.8 + 0.6) / 2 = 0.7
        # subject 2: (0.4 + 0.3) / 2 = 0.35
        #
        # With limit=1, only subject 0 should remain.
        csr1 = csr_array(
            (
                np.array([0.9, 0.8, 0.4]),
                np.array([0, 1, 2]),
                np.array([0, 3]),
            ),
            shape=(1, 3),
        )
        csr2 = csr_array(
            (
                np.array([0.7, 0.6, 0.3]),
                np.array([0, 1, 2]),
                np.array([0, 3]),
            ),
            shape=(1, 3),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr1),
            "source2": SuggestionBatch(csr2),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0), ("source2", 1.0)],
            {"limit": 1},
        )

        assert len(result) == 1

        first_result = next(iter(result))
        suggestions = list(first_result)

        assert len(suggestions) == 1
        assert suggestions[0].score == pytest.approx(0.8)

    def test_train_not_supported(self, project, document_corpus):
        """Test that training is not supported."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )

        with pytest.raises(NotSupportedException):
            backend.train(document_corpus)

    def test_merge_source_batches_single_source_filters_below_threshold(self, project):
        """Test that a single source filters low scores from the final output."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # The source is activated because it has a score >= 0.5.
        # With a single source, low-scoring predictions should be removed
        # from the final result.
        csr = csr_array(
            (
                np.array([0.9, 0.2, 0.1, 0.3]),
                np.array([0, 1, 2, 3]),
                np.array([0, 4]),
            ),
            shape=(1, 4),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0)],
            {"limit": 10},
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            np.array([[0.9, 0.0, 0.0, 0.0]]),
        )

    def test_merge_source_batches_multiple_sources_keep_below_threshold_predictions(
        self, project
    ):
        """Test that multiple sources retain the existing behavior."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        # Both sources are activated because each has a prediction >= 0.5.
        # Low-scoring predictions are still included in the averaging.
        csr1 = csr_array(
            (
                np.array([0.9, 0.2]),
                np.array([0, 1]),
                np.array([0, 2]),
            ),
            shape=(1, 2),
        )
        csr2 = csr_array(
            (
                np.array([0.8, 0.4]),
                np.array([0, 1]),
                np.array([0, 2]),
            ),
            shape=(1, 2),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr1),
            "source2": SuggestionBatch(csr2),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0), ("source2", 1.0)],
            {"limit": 10},
        )

        # Existing multi-source behavior:
        #
        # subject 0: (0.9 + 0.8) / 2 = 0.85
        # subject 1: (0.2 + 0.4) / 2 = 0.30
        np.testing.assert_allclose(
            result.array.toarray(),
            np.array([[0.85, 0.30]]),
        )
