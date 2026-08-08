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
        batch_by_source: Dict[str, List[SuggestionResult]],
        sources: List[Tuple[str, float]],
        params: Dict[str, Any],
    ) -> SuggestionBatch:
        """Merge the given SuggestionBatches from each source into a single
        SuggestionBatch. This implementation computes a weighted
        average based on a Mixture-of-Experts approach that removes batches
        with a top score below a given threshold value."""

        activated = []
        for (project_id, weight), batch in zip(sources, batch_by_source.values()):
            if not batch:
                continue
            # Access the first document's suggestions in the batch
            first_doc_suggestions = batch[0]
            # Check if the suggestions are empty
            if not first_doc_suggestions:
                continue
            # Get the top score from the first document's suggestions
            try:
                top_score = max(score for _, score in first_doc_suggestions)
            except ValueError:
                # Skip if no scores are available
                continue
            if top_score > self.threshold:
                activated.append((batch, top_score))

        if not activated:
            # Return a SuggestionBatch with empty suggestions for each document
            sample_batch = next(iter(batch_by_source.values()))
            num_docs = len(sample_batch)
            empty_matrix = csr_array((num_docs, 0), dtype=float)
            return SuggestionBatch(empty_matrix)

        total_weight = sum(score for _, score in activated)
        weighted_batches = []
        weights = []
        for batch, top_score in activated:
            weight_factor = top_score / total_weight
            weighted_batches.append(batch)
            weights.append(weight_factor)

        return SuggestionBatch.from_averaged(weighted_batches, weights).filter(
            limit=int(params["limit"])
        )