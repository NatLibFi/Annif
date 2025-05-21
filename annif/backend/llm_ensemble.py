"""Backend utilizing a large-language model."""

from __future__ import annotations

import concurrent.futures
import json
import os
from typing import TYPE_CHECKING, Any, Optional

import tiktoken
from openai import AzureOpenAI, BadRequestError, OpenAI, OpenAIError
from transformers import AutoTokenizer

import annif.eval
import annif.parallel
import annif.util
from annif.exception import ConfigurationException, OperationFailedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import backend, ensemble, hyperopt

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
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_base_url = os.getenv("LLM_API_BASE_URL")

        try:
            self.model = self.params["model"]
        except KeyError as err:
            raise ConfigurationException(
                "model setting is missing", project_id=self.project.project_id
            )

        if api_base_url is not None:
            self.client = OpenAI(
                base_url=api_base_url,
                api_key=os.getenv("LLM_API_KEY", "dummy-key"),
            )
        elif azure_endpoint is not None:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=self.params["api_version"],
            )
        else:
            raise OperationFailedException(
                "Please set the AZURE_OPENAI_ENDPOINT or LLM_API_BASE_URL "
                "environment variable for LLM API access."
            )

        # Tokenizer is unnecessary if truncation is not performed
        if int(self.params["max_prompt_tokens"]) > 0:
            self.tokenizer = self._get_tokenizer()
        self._verify_connection()
        super().initialize(parallel)

    def _get_tokenizer(self):
        try:
            # Try OpenAI tokenizer
            base_model = self.model.rsplit("-", 1)[0]
            return tiktoken.encoding_for_model(base_model)
        except KeyError:
            # Fallback to Hugging Face tokenizer
            return AutoTokenizer.from_pretrained(self.model)

    def _verify_connection(self):
        try:
            self._call_llm(
                system_prompt="You are a helpful assistant.",
                prompt="This is a test prompt to verify the connection.",
                params=self.params,
            )
        except OpenAIError as err:
            raise OperationFailedException(
                f"Failed to connect to LLM API: {err}"
            ) from err
        # print(f"Successfully connected to endpoint {self.params['endpoint']}")

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(BaseLLMBackend.DEFAULT_PARAMETERS.copy())
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def _truncate_text(self, text, max_prompt_tokens):
        """Truncate text so it contains at most max_prompt_tokens according to the
        OpenAI tokenizer"""
        tokens = self.tokenizer.encode(text)
        return self.tokenizer.decode(tokens[:max_prompt_tokens])

    def _call_llm(
        self,
        system_prompt: str,
        prompt: str,
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
                model=self.model,
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
        "max_prompt_tokens": 0,
        "llm_weight": 0.7,
        "llm_exponent": 1.0,
        "labels_language": "en",
        "sources_limit": 10,
    }

    def get_hp_optimizer(self, corpus: DocumentCorpus, metric: str) -> None:
        return LLMEnsembleOptimizer(self, corpus, metric)

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        sources = annif.util.parse_sources(params["sources"])
        llm_weight = float(params["llm_weight"])
        llm_exponent = float(params["llm_exponent"])
        if llm_weight < 0.0 or llm_weight > 1.0:
            raise ValueError("llm_weight must be between 0.0 and 1.0")
        if llm_exponent < 0.0:
            raise ValueError("llm_weight_exp must be greater than or equal to 0.0")

        batch_by_source = self._suggest_with_sources(texts, sources)
        merged_source_batch = self._merge_source_batches(
            batch_by_source, sources, {"limit": params["sources_limit"]}
        )

        # Score the suggestion labels with the LLM
        llm_results_batch = self._llm_suggest_batch(texts, merged_source_batch, params)

        batches = [merged_source_batch, llm_results_batch]
        weights = [1.0 - llm_weight, llm_weight]
        exponents = [1.0, llm_exponent]
        return SuggestionBatch.from_averaged(batches, weights, exponents).filter(
            limit=int(params["limit"])
        )

    def _llm_suggest_batch(
        self,
        texts: list[str],
        suggestion_batch: SuggestionBatch,
        params: dict[str, Any],
    ) -> SuggestionBatch:

        max_prompt_tokens = int(params["max_prompt_tokens"])

        system_prompt = """
            You will be given text and a list of keywords to describe it. Your task is
            to score the keywords with a value between 0 and 100. The score value
            should depend on how well the keyword represents the text: a perfect
            keyword should have score 100 and completely unrelated keyword score
            0. You must output JSON with keywords as field names and add their scores
            as field values.
            There must be the same number of objects in the JSON as there are lines in
            the intput keyword list; do not skip scoring any keywords.
        """

        labels_batch = self._get_labels_batch(suggestion_batch)

        def process_single_prompt(text, labels):
            prompt = "Here are the keywords:\n" + "\n".join(labels) + "\n" * 3
            if max_prompt_tokens > 0:
                text = self._truncate_text(text, max_prompt_tokens)
            prompt += "Here is the text:\n" + text + "\n"

            response = self._call_llm(
                system_prompt,
                prompt,
                params,
                response_format={"type": "json_object"},
            )
            try:
                llm_result = json.loads(response)
            except (TypeError, json.decoder.JSONDecodeError) as err:
                print(f"Error decoding JSON response from LLM: {response}")
                print(f"Error: {err}")
                return [SubjectSuggestion(subject_id=None, score=0.0) for _ in labels]

            return [
                (
                    SubjectSuggestion(
                        subject_id=self.project.subjects.by_label(
                            llm_label, self.params["labels_language"]
                        ),
                        score=score / 100.0,  # LLM scores are between 0 and 100
                    )
                    if llm_label in labels
                    else SubjectSuggestion(subject_id=None, score=0.0)
                )
                for llm_label, score in llm_result.items()
            ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            llm_batch_suggestions = list(
                executor.map(process_single_prompt, texts, labels_batch)
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


class LLMEnsembleOptimizer(ensemble.EnsembleOptimizer):
    """Hyperparameter optimizer for the LLM ensemble backend"""

    def _prepare(self, n_jobs=1):
        sources = dict(annif.util.parse_sources(self._backend.params["sources"]))

        # initialize the source projects before forking, to save memory
        # for project_id in sources.keys():
        #     project = self._backend.project.registry.get_project(project_id)
        #     project.initialize(parallel=True)
        self._backend.initialize(parallel=True)

        psmap = annif.parallel.ProjectSuggestMap(
            self._backend.project.registry,
            list(sources.keys()),
            backend_params=None,
            limit=None,
            threshold=0.0,
        )

        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        self._gold_batches = []
        self._source_batches = []

        print("Generating source batches")
        with pool_class(jobs) as pool:
            for suggestions_batch, gold_batch in pool.imap_unordered(
                psmap.suggest_batch, self._corpus.doc_batches
            ):
                self._source_batches.append(suggestions_batch)
                self._gold_batches.append(gold_batch)

        # get the llm batches
        print("Generating LLM batches")
        self._merged_source_batches = []
        self._llm_batches = []
        for batch_by_source, docs_batch in zip(
            self._source_batches, self._corpus.doc_batches
        ):
            merged_source_batch = self._backend._merge_source_batches(
                batch_by_source,
                sources.items(),
                {"limit": self._backend.params["sources_limit"]},
            )
            llm_batch = self._backend._llm_suggest_batch(
                [doc.text for doc in docs_batch],
                merged_source_batch,
                self._backend.params,
            )
            self._merged_source_batches.append(merged_source_batch)
            self._llm_batches.append(llm_batch)

    def _objective(self, trial) -> float:
        eval_batch = annif.eval.EvaluationBatch(self._backend.project.subjects)
        params = {
            "llm_weight": trial.suggest_float("llm_weight", 0.0, 1.0),
            "llm_exponent": trial.suggest_float("llm_exponent", 0.0, 30.0),
        }
        for merged_source_batch, llm_batch, gold_batch in zip(
            self._merged_source_batches, self._llm_batches, self._gold_batches
        ):
            batches = [merged_source_batch, llm_batch]
            weights = [
                1.0 - params["llm_weight"],
                params["llm_weight"],
            ]
            exponents = [
                1.0,
                params["llm_exponent"],
            ]
            avg_batch = SuggestionBatch.from_averaged(
                batches, weights, exponents
            ).filter(limit=int(self._backend.params["limit"]))
            eval_batch.evaluate_many(avg_batch, gold_batch)
        results = eval_batch.results(metrics=[self._metric])
        return results[self._metric]

    def _postprocess(self, study):
        bp = study.best_params
        lines = [
            f"llm_weight={bp['llm_weight']}",
            f"llm_exponent={bp['llm_exponent']}",
        ]
        return hyperopt.HPRecommendation(lines=lines, score=study.best_value)
