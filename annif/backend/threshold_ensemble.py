from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_array

from annif.backend.ensemble import EnsembleBackend
from annif.suggestion import SuggestionBatch

if TYPE_CHECKING:
    from annif.project import AnnifProject


class ThresholdEnsembleBackend(EnsembleBackend):
    def __init__(
        self,
        backend_id: str,
        config_params: Dict[str, Any],
        project: "AnnifProject",
    ):
        self.threshold = float(config_params.get("threshold", 0.1))
        super().__init__(backend_id, config_params, project)

    def _merge_source_batches(
        self,
        batch_by_source: Dict[str, SuggestionBatch],
        sources: List[Tuple[str, float]],
        params: Dict[str, Any],
    ) -> SuggestionBatch:
        """Merge the given SuggestionBatches from each source into a single
        SuggestionBatch. In this implementation, a source is activated for
        a document only if it has at least one suggestion at or above the
        threshold. Among the activated sources, their configured project
        weights determine the weighted average."""

        first_batch = next(iter(batch_by_source.values()))
        n_docs, n_subjects = first_batch.array.shape

        # Accumulate the weighted predictions for every document.
        weighted_sum = csr_array(
            (n_docs, n_subjects),
            dtype="float32",
        )

        # For each document, accumulate the weights of the sources that
        # are actually active for that document.
        weight_sum = np.zeros(n_docs, dtype="float64")

        for project_id, source_weight in sources:
            batch = batch_by_source[project_id]

            if not batch:
                continue

            # Determine which documents have an activated source.
            # The filtered batch is used ONLY for the activation decision.
            filtered_batch = batch.filter(threshold=self.threshold)

            # This gives us one boolean value per document without iterating
            # over SuggestionResult objects in Python.
            active = np.diff(filtered_batch.array.indptr) > 0

            if not np.any(active):
                continue

            # Add this source's configured weight to the denominator for
            # every document for which this source is active.
            weight_sum[active] += source_weight

            # Add this source's original prediction matrix to the numerator,
            # but only for active documents.
            weighted_sum += batch.array.multiply(
                # `active[:, None]` gives one multiplier (0/1) per document:
                active[:, None]
                * source_weight
            ).tocsr()

        # If no source was activated for any document, return an empty
        # SuggestionBatch with the same shape as the source batches.
        if not np.any(weight_sum):
            return SuggestionBatch(csr_array((n_docs, n_subjects), dtype="float32"))

        # Avoid division by zero for documents where no source was active.
        inverse_weight_sum = np.zeros_like(weight_sum)

        # Dividing each row by its corresponding weight_sum gives the same
        # weighted-average calculation as SuggestionBatch.from_averaged(),
        # but with a different set of active sources for each document.
        active_documents = weight_sum > 0
        inverse_weight_sum[active_documents] = 1.0 / weight_sum[active_documents]

        averaged_array = weighted_sum.multiply(inverse_weight_sum[:, None]).tocsr()

        averaged_batch = SuggestionBatch(averaged_array)

        return averaged_batch.filter(limit=int(params.get("limit", 10)))
