"""Backend utilizing a large-language model."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

import tiktoken
from openai import AzureOpenAI, BadRequestError, OpenAIError

import annif.eval
import annif.parallel
import annif.util
from annif.exception import NotSupportedException, OperationFailedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import backend, ensemble

if TYPE_CHECKING:
    from annif.corpus.document import DocumentCorpus


class BaseLLMBackend(backend.AnnifBackend):
    """Base class for LLM backends"""

    DEFAULT_PARAMETERS = {
        "api_version": "2024-10-21",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
    }

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        try:
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=self.params["api_version"],
            )
        except ValueError as err:
            raise OperationFailedException(
                f"Failed to connect to Azure endpoint: {err}"
            ) from err
        self._verify_connection()

    def _verify_connection(self):
        try:
            self._call_llm(
                system_prompt="You are a helpful assistant.",
                prompt="This is a test prompt to verify the connection.",
                model=self.params["model"],
                params=self.params,
            )
        except OpenAIError as err:
            raise OperationFailedException(
                f"Failed to connect to endpoint {self.params['endpoint']}: {err}"
            ) from err
        # print(f"Successfully connected to endpoint {self.params['endpoint']}")

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(BaseLLMBackend.DEFAULT_PARAMETERS.copy())
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def _truncate_text(self, text, encoding, max_prompt_tokens):
        """Truncate text so it contains at most max_prompt_tokens according to the
        OpenAI tokenizer"""
        tokens = encoding.encode(text)
        return encoding.decode(tokens[:max_prompt_tokens])

    def _call_llm(
        self,
        system_prompt: str,
        prompt: str,
        model: str,
        params: dict[str, Any],
        response_format: Optional[dict] = None,
    ) -> str:
        temperature = float(params["temperature"])
        top_p = float(params["top_p"])
        seed = int(params["seed"])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                response_format=response_format,
            )
        except BadRequestError as err:
            print(err)
            return "{}"
        return completion.choices[0].message.content


class LLMEnsembleBackend(BaseLLMBackend, ensemble.EnsembleBackend):
    """Ensemble backend that combines results from multiple projects and scores them
    with a LLM"""

    name = "llm_ensemble"

    DEFAULT_PARAMETERS = {
        "max_prompt_tokens": 127000,
        "llm_weight": 0.7,
        "labels_language": "en",
        "sources_limit": 10,
    }

    def get_hp_optimizer(self, corpus: DocumentCorpus, metric: str) -> None:
        raise NotSupportedException(
            "Hyperparameter optimization for LLM ensemble backend is not possible."
        )

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        sources = annif.util.parse_sources(params["sources"])
        llm_weight = float(params["llm_weight"])
        if llm_weight < 0.0 or llm_weight > 1.0:
            raise ValueError("llm_weight must be between 0.0 and 1.0")

        batch_by_source = self._suggest_with_sources(texts, sources)
        merged_source_batch = self._merge_source_batches(
            batch_by_source, sources, {"limit": params["sources_limit"]}
        )

        # Score the suggestion labels with the LLM
        llm_results_batch = self._llm_suggest_batch(texts, merged_source_batch, params)

        batches = [merged_source_batch, llm_results_batch]
        weights = [1.0 - llm_weight, llm_weight]
        return SuggestionBatch.from_averaged(batches, weights).filter(
            limit=int(params["limit"])
        )

    def _llm_suggest_batch(
        self,
        texts: list[str],
        suggestion_batch: SuggestionBatch,
        params: dict[str, Any],
    ) -> SuggestionBatch:

        model = params["model"]
        encoding = tiktoken.encoding_for_model(model.rsplit("-", 1)[0])
        max_prompt_tokens = int(params["max_prompt_tokens"])

        system_prompt = """
            You will be given text and a list of keywords to describe it. Your task is
            to score the keywords with a value between 0.0 and 1.0. The score value
            should depend on how well the keyword represents the text: a perfect
            keyword should have score 1.0 and completely unrelated keyword score
            0.0. You must output JSON with keywords as field names and add their scores
            as field values.
            There must be the same number of objects in the JSON as there are lines in
            the intput keyword list; do not skip scoring any keywords.
        """

        labels_batch = self._get_labels_batch(suggestion_batch)

        llm_batch_suggestions = []
        for text, labels in zip(texts, labels_batch):
            prompt = "Here are the keywords:\n" + "\n".join(labels) + "\n" * 3
            text = self._truncate_text(text, encoding, max_prompt_tokens)
            prompt += "Here is the text:\n" + text + "\n"

            response = self._call_llm(
                system_prompt,
                prompt,
                model,
                params,
                response_format={"type": "json_object"},
            )
            try:
                llm_result = json.loads(response)
            except (TypeError, json.decoder.JSONDecodeError) as err:
                print(f"Error decoding JSON response from LLM: {response}")
                print(f"Error: {err}")
                llm_batch_suggestions.append(
                    [SubjectSuggestion(subject_id=None, score=0.0) for _ in labels]
                )
                continue
            llm_batch_suggestions.append(
                [
                    (
                        SubjectSuggestion(
                            subject_id=self.project.subjects.by_label(
                                llm_label, self.params["labels_language"]
                            ),
                            score=score,
                        )
                        if llm_label in labels
                        else SubjectSuggestion(subject_id=None, score=0.0)
                    )
                    for llm_label, score in llm_result.items()
                ]
            )

        return SuggestionBatch.from_sequence(
            llm_batch_suggestions,
            self.project.subjects,
        )

    def _get_labels_batch(self, suggestion_batch: SuggestionBatch) -> list[list[str]]:
        return [
            [
                self.project.subjects[suggestion.subject_id].labels[
                    self.params["labels_language"]
                ]
                for suggestion in suggestion_result
            ]
            for suggestion_result in suggestion_batch
        ]
