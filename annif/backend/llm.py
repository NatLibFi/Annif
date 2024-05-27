"""Language model based ensemble backend that combines results from multiple
projects."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import tiktoken
from openai import AzureOpenAI, BadRequestError

import annif.eval
import annif.parallel
import annif.util
from annif.exception import NotSupportedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import backend, ensemble

# from openai import AsyncAzureOpenAI


if TYPE_CHECKING:
    from annif.corpus.document import DocumentCorpus


class BaseLLMBackend(backend.AnnifBackend):
    # """Base class for TODO backends"""

    def initialize(self, parallel: bool = False) -> None:
        # initialize all the source projects
        params = self._get_backend_params(None)
        for project_id, _ in annif.util.parse_sources(params["sources"]):
            project = self.project.registry.get_project(project_id)
            project.initialize(parallel)

        # self.client = AsyncAzureOpenAI(
        self.client = AzureOpenAI(
            azure_endpoint=params["endpoint"],
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
        )


class LLMEnsembleBackend(BaseLLMBackend, ensemble.BaseEnsembleBackend):
    # """TODO backend that combines results from multiple projects"""

    name = "llm_ensemble"

    system_prompt = """
        You will be given text and a list of keywords to describe it. Your task is to
        score the keywords with a value between 0.0 and 1.0. The score value
        should depend on how well the keyword represents the text: a perfect
        keyword should have score 1.0 and completely unrelated keyword score
        0.0. You must output JSON with keywords as field names and add their scores
        as field values.
        There must be the same number of objects in the JSON as there are lines in the
        intput keyword list; do not skip scoring any keywords.
    """
    # Give zero or very low score to the keywords that do not describe the text.

    def _train(self, corpus: DocumentCorpus, params: dict[str, Any], jobs: int = 0):
        raise NotSupportedException("Training LM ensemble backend is not possible.")

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        sources = annif.util.parse_sources(params["sources"])
        batch_by_source = self._suggest_with_sources(texts, sources)
        return self._merge_source_batches(texts, batch_by_source, sources, params)

    def _merge_source_batches(
        self,
        texts: list[str],
        batch_by_source: dict[str, SuggestionBatch],
        sources: list[tuple[str, float]],
        params: dict[str, Any],
    ) -> SuggestionBatch:
        model = params["model"]
        encoding = tiktoken.encoding_for_model(model.rsplit("-", 1)[0])

        batches = [batch_by_source[project_id] for project_id, _ in sources]
        weights = [weight for _, weight in sources]
        avg_sources_suggestion_batch = SuggestionBatch.from_averaged(
            batches, weights
        ).filter(
            limit=int(params["limit"])  # TODO Increase limit
        )

        labels_batch = []
        for suggestionresult in avg_sources_suggestion_batch:
            labels_batch.append(
                [
                    self.project.subjects[s.subject_id].labels[
                        "en"
                    ]  # TODO: make language selectable
                    for s in suggestionresult
                ]
            )

        llm_batch_suggestions = []
        for text, labels in zip(texts, labels_batch):
            prompt = "Here are the keywords:\n" + "\n".join(labels) + "\n" * 3
            text = self._truncate_text(text, encoding)
            prompt += "Here is the text:\n" + text + "\n"

            response = self._call_llm(prompt, model)
            try:
                llm_result = json.loads(response)
            except (TypeError, json.decoder.JSONDecodeError) as err:
                print(err)
                llm_result = None
                continue  # TODO: handle this error

            suggestions = []
            for llm_label, score in llm_result.items():
                subj_id = self.project.subjects.by_label(
                    llm_label, "en"
                )  # TODO: make language selectable
                suggestions.append(SubjectSuggestion(subject_id=subj_id, score=score))
            llm_batch_suggestions.append(suggestions)

        batches.append(
            SuggestionBatch.from_sequence(llm_batch_suggestions, self.project.subjects)
        )
        weights.append(float(params["llm_weight"]))
        return SuggestionBatch.from_averaged(batches, weights).filter(
            limit=int(params["limit"])
        )

    def _truncate_text(self, text, encoding):
        """truncate text so it contains at most MAX_PROMPT_TOKENS according to the
        OpenAI tokenizer"""

        MAX_PROMPT_TOKENS = 14000
        tokens = encoding.encode(text)
        return encoding.decode(tokens[:MAX_PROMPT_TOKENS])

    def _call_llm(self, prompt: str, model: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            completion = self.client.chat.completions.create(
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

            completion = completion.choices[0].message.content
            return completion
        except BadRequestError as err:  # openai.RateLimitError
            print(err)
            return "{}"
