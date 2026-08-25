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
N_SUBJECTS = 2  # Using dummy vocabulary with 2 subjects


def make_batch(scores):
    """Create a SuggestionBatch from a dense score matrix with N_SUBJECTS columns.

    If scores has fewer columns than N_SUBJECTS, it will be padded with zeros.
    """
    scores_array = np.asarray(scores, dtype="float32")
    if scores_array.ndim == 1:
        scores_array = scores_array.reshape(1, -1)

    # Pad with zeros if needed
    if scores_array.shape[1] < N_SUBJECTS:
        padding = np.zeros(
            (scores_array.shape[0], N_SUBJECTS - scores_array.shape[1]), dtype="float32"
        )
        scores_array = np.hstack([scores_array, padding])
    elif scores_array.shape[1] > N_SUBJECTS:
        scores_array = scores_array[:, :N_SUBJECTS]

    return SuggestionBatch(csr_array(scores_array))


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
    def project(self, subject_index):
        """Override the project fixture to use a mock project with N_SUBJECTS."""
        from unittest import mock

        proj = mock.Mock()
        proj.analyzer = annif.analyzer.get_analyzer("snowball(finnish)")
        proj.language = "fi"
        proj.datadir = "/tmp/data"

        # Create a mock subject index with N_SUBJECTS
        from annif.vocab import SubjectIndex

        mock_subjects = mock.Mock(spec=SubjectIndex)
        # Mock the active subjects to have N_SUBJECTS entries
        mock_active = [
            (i, mock.Mock(uri=f"http://example.org/subject{i}"))
            for i in range(N_SUBJECTS)
        ]
        mock_subjects.active = mock_active
        mock_subjects.__len__ = lambda self: N_SUBJECTS
        mock_subjects.by_uri = (
            lambda uri, warnings=False: 0
        )  # Return index 0 for any URI

        proj.subjects = mock_subjects
        proj.vocab = mock.Mock()
        proj.vocab.subjects = mock_subjects
        proj.vocab_lang = "fi"

        # Mock the registry
        mock_registry = mock.Mock()

        # Mock get_project to return a project with N_SUBJECTS for any project_id
        def mock_get_project(project_id, min_access=None):
            mock_proj = mock.Mock()
            mock_proj.subjects = mock_subjects
            return mock_proj

        mock_registry.get_project = mock_get_project
        proj.registry = mock_registry

        return proj

    @pytest.fixture
    def backend(self, project):
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={
                "sources": "dummy",
                "threshold": DEFAULT_THRESHOLD,
            },
            project=project,
        )
        # Mock _align_batch to return the batch as-is for unit tests
        backend._align_batch = lambda batch, project_id: batch
        return backend

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
        """Merging empty batches produces an empty result."""
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
        """When all sources have scores below threshold, no sources are activated.

        Both sources have scores [0.4, 0.3], neither >= 0.5
        No sources are activated, so result is empty
        """
        batches = {
            "source1": make_batch([[0.4, 0.3]]),
            "source2": make_batch([[0.4, 0.3]]),
        }

        result = merge(backend, batches)

        assert len(result) == 1
        assert result.array.nnz == 0

    def test_merge_above_threshold_activates_sources(self, backend):
        """When all sources have scores above threshold, all sources are activated.

        Both sources have scores [0.6, 0.8], both >= 0.5
        Both sources are activated
        Weighted average: [(0.6+0.6)/2, (0.8+0.8)/2] = [0.6, 0.8]
        Highest score is 0.8
        """
        batches = {
            "source1": make_batch([[0.6, 0.8]]),
            "source2": make_batch([[0.6, 0.8]]),
        }

        result = merge(backend, batches)

        assert first_score(result) == pytest.approx(0.8)

    def test_merge_uses_configured_weights(self, backend):
        """Sources with different weights are weighted accordingly.

        source1 (weight=2.0): [0.9]
        source2 (weight=1.0): [0.6]
        Weighted average: (0.9*2 + 0.6*1) / (2+1) = 0.8
        """
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
        """When only one source is activated, its predictions are used directly.

        source1: [0.8, 0.2] - 0.8 >= 0.5, so activated
        source2: [0.1, 0.2] - neither score >= 0.5, so NOT activated
        Result: only source1's predictions: [0.8, 0.2]
        """
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
        """With filter=False (default), below-threshold scores from activated sources
        are preserved in the weighted average.

        Both sources are activated (each has at least one score >= 0.5).
        source1: [0.8, 0.2] - 0.8 >= 0.5, so activated
        source2: [0.1, 0.6] - 0.6 >= 0.5, so activated
        Weighted average with filter=False: [(0.8+0.1)/2, (0.2+0.6)/2] = [0.45, 0.40]
        """
        batches = {
            "source1": make_batch([[0.8, 0.2]]),
            "source2": make_batch([[0.1, 0.6]]),
        }

        result = merge(backend, batches, filter=False)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.45, 0.40]],
        )

    def test_merge_filters_below_threshold_predictions(self, backend):
        """With filter=True, below-threshold scores are removed from activated sources
        before weighted averaging.

        Both sources are activated (each has at least one score >= 0.5).
        source1: [0.8, 0.2] -> filtered: [0.8, 0.0]
        source2: [0.1, 0.6] -> filtered: [0.0, 0.6]
        Weighted average: [(0.8+0.0)/2, (0.0+0.6)/2] = [0.4, 0.3]
        """
        batches = {
            "source1": make_batch([[0.8, 0.2]]),
            "source2": make_batch([[0.1, 0.6]]),
        }

        result = merge(backend, batches, filter=True)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.4, 0.3]],
        )

    def test_merge_applies_threshold_per_document(self, backend):
        """Threshold is applied independently for each document.

        Document 0:
          source1: [0.4] - 0.4 < 0.5, NOT activated
          source2: [0.8] - 0.8 >= 0.5, activated
          Result: [0.8]

        Document 1:
          source1: [0.9] - 0.9 >= 0.5, activated
          source2: [0.3] - 0.3 < 0.5, NOT activated
          Result: [0.9]
        """
        batches = {
            "source1": make_batch([[0.4], [0.9]]),
            "source2": make_batch([[0.8], [0.3]]),
        }

        result = merge(backend, batches)

        assert len(result) == 2
        assert first_score(result, document=0) == pytest.approx(0.8)
        assert first_score(result, document=1) == pytest.approx(0.9)

    def test_merge_recalculates_weights_per_document(self, backend):
        """Weights are recalculated per document based on which sources are active.

        Document 0:
          source1 (weight=2.0): [0.9] - activated
          source2 (weight=1.0): [0.6] - activated
          Weighted average: (0.9*2 + 0.6*1) / (2+1) = 0.8

        Document 1:
          source1 (weight=2.0): [0.4] - NOT activated
          source2 (weight=1.0): [0.8] - activated
          Only source2 is active, so result is [0.8]
        """
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

    def test_single_source_with_filter_true_removes_below_threshold(self, backend):
        """With a single source and filter=True, below-threshold scores are removed.

        source1: [0.9, 0.2] -> filtered: [0.9, 0.0]
        Result: [0.9, 0.0]
        """
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
            filter=True,
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.9, 0.0]],
        )

    def test_single_source_with_filter_false_preserves_below_threshold(self, backend):
        """With a single source and filter=False, below-threshold scores are preserved.

        source1: [0.9, 0.2] - source is activated (0.9 >= 0.5)
        With filter=False, all scores are kept: [0.9, 0.2]
        """
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
            filter=False,
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.9, 0.2]],
        )

    def test_multiple_sources_with_filter_false_keep_below_threshold(
        self,
        backend,
    ):
        """With multiple sources and filter=False (default), below-threshold scores
        from activated sources are preserved.

        Both sources are activated (each has at least one score >= 0.5).
        With filter=False, all scores are kept for averaging.
        Weighted average: [(0.9+0.8)/2, (0.2+0.4)/2] = [0.85, 0.30]
        """
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
            "source2": make_batch([[0.8, 0.4]]),
        }

        result = merge(backend, batches, filter=False)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.85, 0.30]],
        )

    def test_parameter_threshold_overrides_backend_threshold(
        self,
        backend,
    ):
        """Parameter threshold overrides the backend's configured threshold.

        Backend has threshold=0.5, but parameter threshold=0.1
        Both sources have scores >= 0.1, so both are activated
        With filter=False (default), all scores are kept
        Weighted average: [(0.2+0.2)/2, (0.8+0.8)/2] = [0.2, 0.8]
        """
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

    def test_single_source_with_parameter_threshold_and_filter(self, backend):
        """Single source with custom threshold and filter=True
        removes below-threshold scores.

        source1: [0.2, 0.8] with threshold=0.7
        Source is activated (0.8 >= 0.7)
        With filter=True: [0.2, 0.8] -> filtered: [0.0, 0.8]
        Result: [0.0, 0.8]
        """
        batches = {
            "source1": make_batch([[0.2, 0.8]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
            threshold=0.7,
            filter=True,
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

    def test_single_source_high_threshold_with_filter_filters_all(self, backend):
        """With a very high threshold and filter=True, all scores may be removed.

        source1: [0.5, 0.6] with threshold=0.9
        Neither score >= 0.9, so source is NOT activated
        Result: empty batch (no predictions)
        """
        batches = {
            "source1": make_batch([[0.5, 0.6]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
            threshold=0.9,
            filter=True,
        )

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.0, 0.0]],
        )

    def test_single_source_high_threshold_without_filter_preserves_scores(
        self, backend
    ):
        """With a very high threshold and filter=False, source is not activated.

        source1: [0.5, 0.6] with threshold=0.9
        Neither score >= 0.9, so source is NOT activated
        Result: empty batch (no predictions), regardless of filter value
        """
        batches = {
            "source1": make_batch([[0.5, 0.6]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
            threshold=0.9,
            filter=False,
        )

        assert result.array.nnz == 0

    def test_align_batch_returns_batch_when_vocabularies_match(self, project):
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )
        batch = make_batch([[0.9, 0.8]])

        result = ThresholdEnsembleBackend._align_batch(
            backend,
            batch,
            "source1",
        )

        assert result is batch

    def test_align_batch_expands_compact_source_vocabulary(self, project):
        from unittest import mock

        source_subjects = mock.Mock()
        source_subjects.active = [
            (0, mock.Mock(uri="http://example.org/subject0")),
        ]

        project.registry.get_project = lambda project_id: mock.Mock(
            subjects=source_subjects,
        )

        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )
        batch = SuggestionBatch(
            csr_array([[0.9]]),
        )

        result = ThresholdEnsembleBackend._align_batch(
            backend,
            batch,
            "source1",
        )

        assert result.array.shape == (1, N_SUBJECTS)
        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.9, 0.0]],
        )

    def test_merge_skips_empty_source_batch(self, backend):
        batches = {
            "source1": SuggestionBatch(
                csr_array((1, N_SUBJECTS)),
            ),
            "source2": make_batch([[0.8, 0.2]]),
        }

        result = merge(backend, batches)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.8, 0.2]],
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

    def test_optimizer_prepare_adds_backend_and_sources(
        self,
        backend,
        document_corpus,
        monkeypatch,
    ):
        from annif.backend import ensemble

        prepared = {"existing": "value"}

        monkeypatch.setattr(
            ensemble.EnsembleOptimizer,
            "_prepare",
            lambda self, n_jobs=1: prepared,
        )
        monkeypatch.setattr(
            annif.util,
            "parse_sources",
            lambda sources: [("source1", 1.0), ("source2", 2.0)],
        )

        optimizer = ThresholdEnsembleOptimizer(
            backend,
            document_corpus,
            "F1@5",
        )

        result = optimizer._prepare(n_jobs=3)

        assert result == {
            "existing": "value",
            "backend": backend,
            "sources": [("source1", 1.0), ("source2", 2.0)],
        }

    def test_optimizer_postprocess_returns_best_threshold(
        self,
        backend,
        document_corpus,
        capsys,
    ):
        class FakeStudy:
            best_params = {"threshold": 0.123456}
            best_value = 0.87

            def get_trials(self):
                return [
                    type(
                        "Trial",
                        (),
                        {"value": 0.5, "params": {"threshold": 0.1}},
                    )(),
                    type(
                        "Trial",
                        (),
                        {"value": 0.7, "params": {"threshold": 0.2}},
                    )(),
                    type(
                        "Trial",
                        (),
                        {"value": None, "params": {"threshold": 0.3}},
                    )(),
                    type(
                        "Trial",
                        (),
                        {"value": 0.9, "params": {"other": 1.0}},
                    )(),
                    type(
                        "Trial",
                        (),
                        {"value": 0.2, "params": {"threshold": 0.0}},
                    )(),
                ]

        optimizer = ThresholdEnsembleOptimizer(
            backend,
            document_corpus,
            "F1@5",
        )

        recommendation = optimizer._postprocess(FakeStudy())

        assert recommendation.lines == ["threshold=0.1235"]
        assert recommendation.score == 0.87

        output = capsys.readouterr().out
        assert "Found isoelastic point with score" in output
        assert "threshold=" in output

    def test_calculate_isoelastic_point(self, backend, document_corpus):
        class FakeTrial:
            def __init__(self, threshold, value):
                self.params = {"threshold": threshold}
                self.value = value

        class FakeStudy:
            def get_trials(self):
                return [
                    FakeTrial(0.1, 0.5),
                    FakeTrial(0.2, 0.7),
                    FakeTrial(0.4, 0.9),
                ]

        optimizer = ThresholdEnsembleOptimizer(
            backend,
            document_corpus,
            "F1@5",
        )

        isoelastic_x, isoelastic_y = optimizer.calculate_isoelastic_point(
            FakeStudy(),
        )

        assert isoelastic_x == pytest.approx(0.2885390082)
        assert isoelastic_y == pytest.approx(0.8057532746)

    def test_init_default_filter(self, project):
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={"sources": "dummy"},
            project=project,
        )

        assert backend.filter is False

    def test_init_custom_filter_true(self, project):
        backend = ThresholdEnsembleBackend(
            backend_id="threshold_ensemble",
            config_params={
                "sources": "dummy",
                "filter": True,
            },
            project=project,
        )

        assert backend.filter is True

    def test_merge_with_filter_false_keeps_below_threshold_scores(self, backend):
        backend.filter = False
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
        }

        result = merge(backend, batches, sources=[("source1", 1.0)], filter=False)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.9, 0.2]],
        )

    def test_merge_with_filter_true_removes_below_threshold_scores(self, backend):
        backend.filter = False
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
        }

        result = merge(backend, batches, sources=[("source1", 1.0)], filter=True)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.9, 0.0]],
        )

    def test_merge_with_multiple_sources_filter_false_preserves_below_threshold(
        self, backend
    ):
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
            "source2": make_batch([[0.8, 0.4]]),
        }

        result = merge(backend, batches, filter=False)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.85, 0.30]],
        )

    def test_merge_with_multiple_sources_filter_true_filters_below_threshold(
        self, backend
    ):
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
            "source2": make_batch([[0.8, 0.4]]),
        }

        result = merge(backend, batches, filter=True)

        # With filter=True, both sources have at least one score >= threshold (0.5)
        # source1: [0.9, 0.2] -> filtered: [0.9, 0.0]
        # source2: [0.8, 0.4] -> filtered: [0.8, 0.0]
        # Weighted average: [(0.9 + 0.8) / 2, (0.0 + 0.0) / 2] = [0.85, 0.0]
        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.85, 0.0]],
        )

    def test_filter_parameter_overrides_backend_config(self, backend):
        backend.filter = False
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
        }

        # Parameter filter=True should override backend.filter=False
        result = merge(backend, batches, sources=[("source1", 1.0)], filter=True)

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.9, 0.0]],
        )

    def test_backend_config_filter_used_when_parameter_missing(self, backend):
        backend.filter = True
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
        }

        # Should use backend.filter=True since parameter is not provided
        result = merge(backend, batches, sources=[("source1", 1.0)])

        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.9, 0.0]],
        )

    def test_weighted_sources_with_filter_true(self, backend):
        """Test that weighted averaging works correctly with filter=True.

        source1 (weight=2.0): [0.9, 0.2] -> filtered: [0.9, 0.0]
        source2 (weight=1.0): [0.8, 0.4] -> filtered: [0.8, 0.0]
        Weighted average: [(0.9*2 + 0.8*1)/(2+1),
                          (0.0*2 + 0.0*1)/(2+1)] = [0.866..., 0.0]
        """
        batches = {
            "source1": make_batch([[0.9, 0.2]]),
            "source2": make_batch([[0.8, 0.4]]),
        }

        result = merge(
            backend,
            batches,
            sources=[("source1", 2.0), ("source2", 1.0)],
            filter=True,
        )

        expected_weighted_avg = (0.9 * 2 + 0.8 * 1) / (2 + 1)
        np.testing.assert_allclose(
            result.array.toarray(),
            [[expected_weighted_avg, 0.0]],
        )

    def test_threshold_exact_match(self, backend):
        """Test that scores exactly equal to the threshold are kept.

        With threshold=0.5, a score of exactly 0.5 should activate the source.
        """
        batches = {
            "source1": make_batch([[0.5, 0.2]]),
        }

        # Score of 0.5 should be >= threshold, so source is activated
        result = merge(
            backend,
            batches,
            sources=[("source1", 1.0)],
            filter=True,
        )

        # With filter=True, 0.5 is kept (>= threshold), 0.2 is removed
        np.testing.assert_allclose(
            result.array.toarray(),
            [[0.5, 0.0]],
        )

    def test_all_scores_below_threshold(self, backend):
        """Test that when all scores are below threshold, no source is activated."""
        batches = {
            "source1": make_batch([[0.4, 0.3]]),
            "source2": make_batch([[0.2, 0.1]]),
        }

        result = merge(backend, batches, filter=False)

        # No source has any score >= 0.5, so no sources are activated
        assert result.array.nnz == 0

    def test_make_batch_padding(self):
        """Test that make_batch correctly pads scores to N_SUBJECTS columns."""
        # Test with 1 column (should be padded to 2)
        batch = make_batch([[0.9]])
        assert batch.array.shape == (1, N_SUBJECTS)
        np.testing.assert_allclose(
            batch.array.toarray(),
            [[0.9, 0.0]],
        )

        # Test with 3 columns (should be truncated to 2)
        batch = make_batch([[0.9, 0.8, 0.7]])
        assert batch.array.shape == (1, N_SUBJECTS)
        np.testing.assert_allclose(
            batch.array.toarray(),
            [[0.9, 0.8]],
        )
