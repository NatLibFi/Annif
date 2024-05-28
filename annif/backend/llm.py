"""TODO"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from openai import AzureOpenAI, BadRequestError

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


class BaseLLMBackend(backend.AnnifBackend):
    # """Base class for TODO backends"""

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

    def _suggest_with_sources(
        self, texts: list[str], sources: list[tuple[str, float]]
    ) -> dict[str, SuggestionBatch]:
        return {
            project_id: self.project.registry.get_project(project_id).suggest(texts)
            for project_id, _ in sources
        }

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        sources = annif.util.parse_sources(params["sources"])
        return self._suggest_with_sources(texts, sources)[sources[0][0]]
        # return self._merge_source_batches(batch_by_source, sources, params)


class LLMBackend(BaseLLMBackend):
    # """TODO backend that combines results from multiple projects"""

    name = "llm"

    system_prompt = """
        You will be given text and a list of keywords to describe it. Your task is to
        score the keywords with a value between 0.0 and 1.0. The score value
        should depend on how well the keyword represents the text: a perfect
        keyword should have score 1.0 and completely unrelated keyword score
        0.0. You must output JSON with keywords as field names and add their scores
        as field values.
        There must be the same number of items in the JSON as there are in the
        intput keyword list.
    """

    @property
    def is_trained(self) -> bool:
        sources_trained = self._get_sources_attribute("is_trained")
        return all(sources_trained)

    @property
    def modification_time(self) -> datetime | None:
        mtimes = self._get_sources_attribute("modification_time")
        return max(filter(None, mtimes), default=None)

    def _train(self, corpus: DocumentCorpus, params: dict[str, Any], jobs: int = 0):
        raise NotSupportedException("Training LLM backend is not possible.")

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        sources = annif.util.parse_sources(params["sources"])
        endpoint = params["endpoint"]
        model = params["model"]
        chars_max = 40000

        batch_results = []
        base_suggestion_batch = self._suggest_with_sources(texts, sources)[
            sources[0][0]
        ]

        for text, base_suggestions in zip(texts, base_suggestion_batch):
            prompt = "Here is the text:\n" + text[:chars_max] + "\n"

            base_labels = [
                self.project.subjects[s.subject_id].labels["en"]
                for s in base_suggestions
            ]
            prompt += "And here are the keywords:\n" + "\n".join(base_labels)
            answer = self._call_llm(prompt, endpoint, model)
            try:
                llm_result = json.loads(answer)
            except TypeError as err:
                print(err)
                llm_result = dict()
            results = self._get_llm_suggestions(
                llm_result, base_labels, base_suggestions
            )
            batch_results.append(results)
        return SuggestionBatch.from_sequence(batch_results, self.project.subjects)

    def _get_llm_suggestions(self, llm_result, base_labels, base_suggestions):
        suggestions = []
        print(f"LLM result: {llm_result}")
        for blabel, bsuggestion in zip(base_labels, base_suggestions):
            # score = llm_result.get(blabel, 0)
            try:
                score = llm_result[blabel]
            except KeyError:
                print(f"Base label {blabel} not found in LLM labels")
                score = 0  # bsuggestion.score
            subj_id = bsuggestion.subject_id
            mean_score = (bsuggestion.score + score) / 2  # Mean of LLM and base scores!
            suggestions.append(SubjectSuggestion(subject_id=subj_id, score=mean_score))
        return suggestions

    # async def _call_llm(self, prompt: str, endpoint: str, model: str):
    def _call_llm(self, prompt: str, endpoint: str, model: str):
        # client = AsyncAzureOpenAI(
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            # completion = await client.chat.completions.create(
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                seed=0,
                max_tokens=1800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content
        except BadRequestError as err:
            print(err)
            return "{}"
