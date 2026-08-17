"""Unit tests for the ThresholdEnsembleBackend in Annif"""

import numpy as np
import pytest
from scipy.sparse import csr_array

import annif.eval
from annif.backend.threshold_ensemble import (
    ThresholdEnsembleBackend,
    ThresholdEnsembleHPObjective,
    ThresholdEnsembleOptimizer,
)
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

    def test_merge_source_batches_uses_parameter_threshold(self, project):
        """Test that an optimizer-supplied threshold overrides the backend threshold."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        data = np.array([0.2, 0.8])
        indices = np.array([0, 1])
        indptr = np.array([0, 2])

        csr = csr_array(
            (data, indices, indptr),
            shape=(1, 2),
        )

        batch = SuggestionBatch(csr)

        result = backend._merge_source_batches(
            {"source1": batch, "source2": batch},
            [("source1", 1.0), ("source2", 1.0)],
            {
                "threshold": 0.1,
                "limit": 10,
            },
        )

        assert len(result) == 1

        result_array = result.array.toarray()

        # Both sources are active at threshold 0.1, so their original
        # predictions are averaged. The 0.2 prediction is retained.
        np.testing.assert_allclose(
            result_array,
            [[0.2, 0.8]],
        )

    def test_single_source_uses_parameter_threshold(self, project):
        """Test that the supplied threshold is used for a single source."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.5},
            project=project,
        )

        data = np.array([0.2, 0.8])
        indices = np.array([0, 1])
        indptr = np.array([0, 2])

        csr = csr_array(
            (data, indices, indptr),
            shape=(1, 2),
        )

        result = backend._merge_source_batches(
            {"source1": SuggestionBatch(csr)},
            [("source1", 1.0)],
            {
                "threshold": 0.7,
                "limit": 10,
            },
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.0, 0.8]],
        )

    def test_hp_objective_uses_trial_threshold(self, monkeypatch):
        """Test that the Optuna threshold is passed to the backend."""

        class FakeTrial:
            def suggest_float(self, name, low, high, log=False):
                assert name == "threshold"
                assert low == 0.001
                assert high == 0.5
                assert log is True
                return 0.123

        class FakeEvaluationBatch:
            instance = None

            def __init__(self, subject_index):
                self.calls = []
                FakeEvaluationBatch.instance = self

            def evaluate_many(self, merged_batch, gold_batch):
                self.calls.append((merged_batch, gold_batch))

            def results(self, metrics):
                assert metrics == ["F1@5"]
                return {"F1@5": 0.42}

        class FakeBackend:
            def __init__(self):
                self.calls = []

            def _merge_source_batches(self, source_batches, sources, params):
                self.calls.append((source_batches, sources, params))
                return "merged"

        monkeypatch.setattr(
            annif.eval,
            "EvaluationBatch",
            FakeEvaluationBatch,
        )

        backend = FakeBackend()

        args = {
            "backend": backend,
            "gold_batches": ["gold"],
            "source_batches": ["sources"],
            "subject_index": "subjects",
            "sources": ["source1", "source2"],
            "limit": 100,
            "metric": "F1@5",
        }

        result = ThresholdEnsembleHPObjective.objective(
            FakeTrial(),
            args,
        )

        assert result == 0.42
        assert backend.calls == [
            (
                "sources",
                ["source1", "source2"],
                {
                    "limit": 100,
                    "threshold": 0.123,
                },
            )
        ]

    def test_hp_objective_threshold_is_used_for_single_source(self, project):
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={
                "sources": "source1",
                "threshold": 0.1,
            },
            project=project,
        )

        csr = csr_array(
            (
                np.array([0.08, 0.12]),
                np.array([0, 1]),
                np.array([0, 2]),
            ),
            shape=(1, 2),
        )

        batch = SuggestionBatch(csr)

        result = backend._merge_source_batches(
            {"source1": batch},
            [("source1", 1.0)],
            {"threshold": 0.05, "limit": 10},
        )

        # The trial threshold (0.05), not the configured threshold (0.1),
        # should be used.
        assert result.array.nnz == 2

    def test_merge_source_batches_empty_sources(self, project):
        """Test merging with an empty sources list."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )

        # Create non-empty batches
        csr = csr_array(
            (np.array([0.6, 0.8]), np.array([0, 1]), np.array([0, 2])),
            shape=(1, 2),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr),
        }

        # Empty sources list
        result = backend._merge_source_batches(
            batch_by_source,
            [],  # Empty sources
            {"limit": 10},
        )

        # Should return a batch with same shape as input but no suggestions
        assert isinstance(result, SuggestionBatch)
        assert len(result) == 1  # Same number of documents as input
        assert result.array.nnz == 0  # But no active suggestions

    def test_merge_source_batches_threshold_zero(self, project):
        """Test that threshold=0.0 activates all sources."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.0},
            project=project,
        )

        # Even very low scores should activate sources
        data = np.array([0.01, 0.02])
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

        # Both sources should be activated
        assert len(result) == 1
        result_list = list(result)
        assert len(result_list) > 0
        first_suggestions = list(result_list[0])
        # Should have averaged predictions
        assert len(first_suggestions) == 2

    def test_merge_source_batches_threshold_very_high(self, project):
        """Test that a very high threshold deactivates all sources."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 10.0},
            project=project,
        )

        # Scores are all below 10.0
        data = np.array([0.5, 0.8])
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

        # No sources should be activated, result should be empty
        assert isinstance(result, SuggestionBatch)
        assert result.array.nnz == 0

    def test_merge_source_batches_limit_zero(self, project):
        """Test that limit=0 returns an empty batch."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.1},
            project=project,
        )

        data = np.array([0.6, 0.8])
        indices = np.array([0, 1])
        indptr = np.array([0, 2])

        csr = csr_array(
            (data, indices, indptr),
            shape=(1, 2),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr),
            "source2": SuggestionBatch(csr),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0), ("source2", 1.0)],
            {"limit": 0},
        )

        # Should return empty batch
        assert isinstance(result, SuggestionBatch)
        assert result.array.nnz == 0

    def test_merge_source_batches_single_source_high_threshold(self, project):
        """Test single source with high threshold filters all predictions."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.9},
            project=project,
        )

        # All scores below 0.9
        csr = csr_array(
            (np.array([0.5, 0.6, 0.7]), np.array([0, 1, 2]), np.array([0, 3])),
            shape=(1, 3),
        )

        batch_by_source = {
            "source1": SuggestionBatch(csr),
        }

        result = backend._merge_source_batches(
            batch_by_source,
            [("source1", 1.0)],
            {"limit": 10},
        )

        # All predictions should be filtered out
        np.testing.assert_allclose(
            result.array.toarray(),
            np.array([[0.0, 0.0, 0.0]]),
        )

    def test_get_hp_optimizer(self, project, document_corpus):
        """Test that get_hp_optimizer returns the correct optimizer type."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )

        optimizer = backend.get_hp_optimizer(document_corpus, "F1@5")

        assert isinstance(optimizer, ThresholdEnsembleOptimizer)
        assert optimizer._backend is backend
        assert optimizer._corpus is document_corpus
        assert optimizer._metric == "F1@5"

    def test_merge_source_batches_uses_config_threshold_when_param_missing(
        self, project
    ):
        """Test that config threshold is used when param doesn't specify threshold."""
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy", "threshold": 0.3},
            project=project,
        )

        data = np.array([0.2, 0.8])
        indices = np.array([0, 1])
        indptr = np.array([0, 2])

        csr = csr_array(
            (data, indices, indptr),
            shape=(1, 2),
        )

        batch = SuggestionBatch(csr)

        result = backend._merge_source_batches(
            {"source1": batch, "source2": batch},
            [("source1", 1.0), ("source2", 1.0)],
            {"limit": 10},  # No threshold in params
        )

        # With threshold 0.3 from config, both sources should be activated
        # because 0.8 > 0.3 (even though 0.2 < 0.3, the source is activated
        # if ANY prediction >= threshold)
        # The averaged result should be [0.2, 0.8] from original arrays
        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.2, 0.8]],  # Both sources activated, original scores averaged
        )
