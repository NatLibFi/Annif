from typing import Any, Dict, Tuple, List
from annif.backend.ensemble import EnsembleBackend
from scipy.sparse import csr_array
from annif.suggestion import SuggestionBatch, SuggestionResult

class MoEEnsembleBackend(EnsembleBackend):
    def __init__(
        self,
        backend_id: str,
        config_params: Dict[str, Any],
        project: "AnnifProject",
    ):
        self.threshold = float(config_params.get('threshold', 0.5))
        super().__init__(backend_id, config_params, project)

    def _merge_source_batches(
        self,
        batch_by_source: Dict[str, SuggestionBatch],
        sources: List[Tuple[str, float]],
        params: Dict[str, Any],
    ) -> SuggestionBatch:
        """Merge the given SuggestionBatches from each source into a single
        SuggestionBatch. This implementation computes a weighted
        average based on a Mixture-of-Experts approach that removes batches
        with a top score below a given threshold value."""

        activated = []
        for project_id, batch in batch_by_source.items():
            if not batch:
                continue

            # Iterate over the first document's suggestions in the batch
            first_doc = next(iter(batch), None)
            if not first_doc:
                continue

            # Get the top score from the first document's suggestions
            top_score = max(score for _, score in first_doc) if first_doc else 0.0

            if top_score > self.threshold:
                activated.append((batch, top_score))

        if not activated:
            # Return an empty SuggestionBatch with an empty csr_array
            return SuggestionBatch(csr_array((0, 0)))

        total_weight = sum(score for _, score in activated)
        weighted_batches = []
        weights = []
        for batch, top_score in activated:
            weight_factor = top_score / total_weight
            weighted_batches.append(batch)
            weights.append(weight_factor)

        # Combine batches using weighted averaging
        averaged_batch = SuggestionBatch.from_averaged(weighted_batches, weights)
        return averaged_batch.filter(limit=int(params.get("limit", 10)))