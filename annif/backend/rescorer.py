"""TODO"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

# import tiktoken
from transformers import pipeline

import annif.eval
import annif.parallel
import annif.util
from annif.exception import NotSupportedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import backend

# from openai import AsyncAzureOpenAI


if TYPE_CHECKING:
    from datetime import datetime

    from annif.corpus.document import DocumentCorpus


class RescorerBackend(backend.AnnifBackend):
    # """TODO backend that combines results from multiple projects"""

    name = "rescorer"

    def _get_sources_attribute(self, attr: str) -> list[bool | None]:
        params = self._get_backend_params(None)
        sources = annif.util.parse_sources(params["sources"])
        return [
            getattr(self.project.registry.get_project(project_id), attr)
            for project_id, _ in sources
        ]

    def initialize(self, parallel: bool = False) -> None:
        # initialize all the source projects
        params = self._get_backend_params(None)
        for project_id, _ in annif.util.parse_sources(params["sources"]):
            project = self.project.registry.get_project(project_id)
            project.initialize(parallel)

        self.classifier = pipeline(
            "zero-shot-classification", model=params.get("model"),
            from_pt=True,
            multi_label=True,
        )

    def _suggest_with_sources(
        self, texts: list[str], sources: list[tuple[str, float]]
    ) -> dict[str, SuggestionBatch]:
        return {
            project_id: self.project.registry.get_project(project_id).suggest(texts)
            for project_id, _ in sources
        }

    @property
    def is_trained(self) -> bool:
        sources_trained = self._get_sources_attribute("is_trained")
        return all(sources_trained)

    @property
    def modification_time(self) -> datetime | None:
        mtimes = self._get_sources_attribute("modification_time")
        return max(filter(None, mtimes), default=None)

    def _train(self, corpus: DocumentCorpus, params: dict[str, Any], jobs: int = 0):
        raise NotSupportedException("Training rescorer backend is not possible.")

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        sources = annif.util.parse_sources(params["sources"])
        new_scores_weight = float(params["new_scores_weight"])
        # llm_probs_weight = float(params["llm_probs_weight"])
        # encoding = tiktoken.encoding_for_model(model.rsplit("-", 1)[0])

        batch_results = []
        base_suggestion_batch = self._suggest_with_sources(texts, sources)[
            sources[0][0]
        ]

        from time import time
        start_t = time()
        for text, base_suggestions in zip(texts, base_suggestion_batch):
            base_labels = [
                self.project.subjects[s.subject_id].labels["en"]
                for s in base_suggestions
            ]

            # text = self._truncate_text(text, encoding)
            result = self.classifier(text, base_labels)
            print(result)
            # try:
            #    llm_result = json.loads(answer)
            # except (TypeError, json.decoder.JSONDecodeError) as err:
            #    print(err)
            #    llm_result = dict()
            rescored_results = self._rescore_suggestions(
                result,
                base_labels,
                base_suggestions,
                new_scores_weight,
            )
            batch_results.append(rescored_results)
        print(f"Time: {time() - start_t:.2f} s")
        return SuggestionBatch.from_sequence(batch_results, self.project.subjects)

    # def _truncate_text(self, text, encoding):
    #     """truncate text so it contains at most MAX_PROMPT_TOKENS according to the
    #     OpenAI tokenizer"""

    #     MAX_PROMPT_TOKENS = 14000
    #     tokens = encoding.encode(text)
    #     return encoding.decode(tokens[:MAX_PROMPT_TOKENS])

    def _rescore_suggestions(
        self,
        result,
        base_labels,
        base_suggestions,
        new_scores_weight,
    ):
        suggestions = []
        for blabel, bsuggestion in zip(base_labels, base_suggestions):
            try:
                ind = result["labels"].index(blabel)
                score = result["scores"][ind]
            except ValueError:
                print(f"Base label {blabel} not found in new labels")
                score = bsuggestion.score  # use only base suggestion score
            subj_id = bsuggestion.subject_id

            base_scores_weight = 1.0 - new_scores_weight
            mean_score = (
                base_scores_weight * bsuggestion.score
                + new_scores_weight * score  # * probability * llm_probs_weight
            ) / (
                base_scores_weight
                + new_scores_weight  # * probability * llm_probs_weight
            )  # weighted mean of LLM and base scores!
            suggestions.append(SubjectSuggestion(subject_id=subj_id, score=mean_score))
        return suggestions
