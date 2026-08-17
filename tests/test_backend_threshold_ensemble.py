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

SOURCES = [("source1", 1.0), ("source2", 1.0)]
DEFAULT_LIMIT = 10
DEFAULT_THRESHOLD = 0.5


def make_batch(scores):
    """Create a SuggestionBatch from a dense score matrix."""
    return SuggestionBatch(csr_array(np.asarray(scores, dtype="float32")))


def merge(backend, batches, sources=SOURCES, **params):
    """Merge source batches using the supplied parameters."""
    return backend._merge_source_batches(
        batches,
        sources,
        {"limit": DEFAULT_LIMIT, **params},
    )


def first_score(batch, document=0):
    """Return the highest-scoring suggestion for a document."""
    return next(iter(list(batch)[document])).score


class TestThresholdEnsembleBackend:
    @pytest.fixture
    def backend(self, project):
        return ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={
                "sources": "dummy",
                "threshold": DEFAULT_THRESHOLD,
            },
            project=project,
        )

    def test_init_default_threshold(self, project):
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )

        assert backend.threshold == 0.1

    def test_init_custom_threshold(self, project):
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={
                "sources": "dummy",
                "threshold": 0.7,
            },
            project=project,
        )

        assert backend.threshold == 0.7

    def test_merge_empty_batches(self, project):
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )
        empty = make_batch(np.empty((0, 0)))

        result = merge(
            backend,
            {"source1": empty, "source2": empty},
        )

        assert isinstance(result, SuggestionBatch)
        assert len(result) == 0

    def test_merge_below_threshold_produces_no_suggestions(self, backend):
        batches = {
            "source1": make_batch([[0.4, 0.3]]),
            "source2": make_batch([[0.4, 0.3]]),
        }

        result = merge(backend, batches)

        assert len(result) == 1
        assert result.array.nnz == 0

    def test_merge_above_threshold_activates_sources(self, backend):
        batches = {
            "source1": make_batch([[0.6, 0.8]]),
            "source2": make_batch([[0.6, 0.8]]),
        }

        result = merge(backend, batches)

        assert first_score(result) == pytest.approx(0.8)

    def test_merge_uses_configured_weights(self, backend):
        batches = {
            "source1": make_batch([[0.9]]),
            "source2": make_batch([[0.6]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 2.0), ("source2", 1.0)],
        )

        assert first_score(result) == pytest.approx(0.8)

    def test_merge_single_active_source(self, backend):
        batches = {
            "source1": make_batch([[0.8, 0.2]]),
            "source2": make_batch([[0.1, 0.2]]),
        }

        result = merge(backend, batches)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.8, 0.2]],
        )

    def test_merge_preserves_below_threshold_predictions(self, backend):
        batches = {
            "source1": make_batch([[0.8, 0.2]]),
            "source2": make_batch([[0.1, 0.6]]),
        }

        result = merge(backend, batches)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.45, 0.40]],
        )

    def test_merge_applies_threshold_per_document(self, backend):
        batches = {
            "source1": make_batch([[0.4], [0.9]]),
            "source2": make_batch([[0.8], [0.3]]),
        }

        result = merge(backend, batches)

        assert len(result) == 2
        assert first_score(result, document=0) == pytest.approx(0.8)
        assert first_score(result, document=1) == pytest.approx(0.9)

    def test_merge_recalculates_weights_per_document(self, backend):
        batches = {
            "source1": make_batch([[0.9], [0.4]]),
            "source2": make_batch([[0.6], [0.8]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 2.0), ("source2", 1.0)],
        )

        assert first_score(result, document=0) == pytest.approx(0.8)
        assert first_score(result, document=1) == pytest.approx(0.8)

    def test_merge_applies_limit_after_averaging(self, backend):
        batches = {
            "source1": make_batch([[0.9, 0.8, 0.4]]),
            "source2": make_batch([[0.7, 0.6, 0.3]]),
        }

        result = merge(backend, batches, limit=1)

        suggestions = list(next(iter(result)))

        assert len(suggestions) == 1
        assert suggestions[0].score == pytest.approx(0.8)

    def test_train_not_supported(self, backend, document_corpus):
        with pytest.raises(NotSupportedException):
            backend.train(document_corpus)

    def test_single_source_filters_below_threshold(self, backend):
        batches = {
            "source1": make_batch([[0.9, 0.2, 0.1, 0.3]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.9, 0.0, 0.0, 0.0]],
        )

    def test_multiple_sources_keep_below_threshold_predictions(
        self,
        backend,
    ):
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
            "source2": make_batch([[0.8, 0.4]]),
        }

        result = merge(backend, batches)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.85, 0.30]],
        )

    def test_parameter_threshold_overrides_backend_threshold(
        self,
        backend,
    ):
        batches = {
            "source1": make_batch([[0.2, 0.8]]),
            "source2": make_batch([[0.2, 0.8]]),
        }

        result = merge(
            backend,
            batches,
            threshold=0.1,
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.2, 0.8]],
        )

    def test_single_source_uses_parameter_threshold(self, backend):
        batches = {
            "source1": make_batch([[0.2, 0.8]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
            threshold=0.7,
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.0, 0.8]],
        )

    def test_config_threshold_used_when_parameter_missing(self, project):
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={
                "sources": "dummy",
                "threshold": 0.3,
            },
            project=project,
        )
        batches = {
            "source1": make_batch([[0.2, 0.8]]),
            "source2": make_batch([[0.2, 0.8]]),
        }

        result = merge(backend, batches)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.2, 0.8]],
        )

    def test_hp_objective_uses_trial_threshold(self, monkeypatch):
        class FakeTrial:
            def suggest_float(self, name, low, high, log=False):
                assert name == "threshold"
                assert low == 0.001
                assert high == 0.5
                assert log is True
                return 0.123

        class FakeEvaluationBatch:
            def __init__(self, subject_index):
                self.calls = []

            def evaluate_many(self, merged_batch, gold_batch):
                self.calls.append((merged_batch, gold_batch))

            def results(self, metrics):
                assert metrics == ["F1@5"]
                return {"F1@5": 0.42}

        class FakeBackend:
            def __init__(self):
                self.calls = []

            def _merge_source_batches(
                self,
                source_batches,
                sources,
                params,
            ):
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

    def test_hp_threshold_can_be_used_for_single_source(self, backend):
        batches = {
            "source1": make_batch([[0.08, 0.12]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
            threshold=0.05,
        )

        assert result.array.nnz == 2

    def test_merge_empty_sources(self, backend):
        batches = {
            "source1": make_batch([[0.6, 0.8]]),
        }

        result = backend._merge_source_batches(
            batches,
            [],
            {"limit": DEFAULT_LIMIT},
        )

        assert isinstance(result, SuggestionBatch)
        assert len(result) == 1
        assert result.array.nnz == 0

    def test_threshold_zero_activates_all_sources(self, backend):
        batches = {
            "source1": make_batch([[0.01, 0.02]]),
            "source2": make_batch([[0.01, 0.02]]),
        }

        result = merge(
            backend,
            batches,
            threshold=0.0,
        )

        suggestions = list(next(iter(result)))

        assert len(suggestions) == 2

    def test_very_high_threshold_deactivates_all_sources(self, backend):
        batches = {
            "source1": make_batch([[0.5, 0.8]]),
            "source2": make_batch([[0.5, 0.8]]),
        }

        result = merge(
            backend,
            batches,
            threshold=10.0,
        )

        assert isinstance(result, SuggestionBatch)
        assert result.array.nnz == 0

    def test_limit_zero_returns_empty_batch(self, backend):
        batches = {
            "source1": make_batch([[0.6, 0.8]]),
            "source2": make_batch([[0.6, 0.8]]),
        }

        result = merge(
            backend,
            batches,
            limit=0,
        )

        assert isinstance(result, SuggestionBatch)
        assert result.array.nnz == 0

    def test_single_source_high_threshold_filters_all_predictions(
        self,
        backend,
    ):
        batches = {
            "source1": make_batch([[0.5, 0.6, 0.7]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
            threshold=0.9,
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.0, 0.0, 0.0]],
        )

    def test_get_hp_optimizer(self, backend, document_corpus):
        optimizer = backend.get_hp_optimizer(
            document_corpus,
            "F1@5",
        )

        assert isinstance(optimizer, ThresholdEnsembleOptimizer)
        assert optimizer._backend is backend
        assert optimizer._corpus is document_corpus
        assert optimizer._metric == "F1@5"
