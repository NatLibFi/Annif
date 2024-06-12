"""TODO"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import tiktoken
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

        # self.client = AsyncAzureOpenAI(
        self.client = AzureOpenAI(
            azure_endpoint=params["endpoint"],
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
        )

    def _suggest_with_sources(
        self, texts: list[str], sources: list[tuple[str, float]]
    ) -> dict[str, SuggestionBatch]:
        return {
            project_id: self.project.registry.get_project(project_id).suggest(texts)
            for project_id, _ in sources
        }


class LLMBackend(BaseLLMBackend):
    # """TODO backend that combines results from multiple projects"""

    name = "llm"

    system_prompt = """
        You will be given text and a list of keywords to describe it. Your task is to
        decide whether a keyword is suitable for the text and describes it well:
        give output as a binary value; 1 for good keywords and 0 for keywords that do
        not describe the text. You must output JSON with keywords as field names and
        the binary scores as field values.
        There must be the same number of items in the JSON as there are in the
        intput keyword list, so give either 0 or 1 to every input keyword.
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
        model = params["model"]
        llm_scores_weight = float(params["llm_scores_weight"])
        # llm_probs_weight = float(params["llm_probs_weight"])
        encoding = tiktoken.encoding_for_model(model.rsplit("-", 1)[0])

        batch_results = []
        base_suggestion_batch = self._suggest_with_sources(texts, sources)[
            sources[0][0]
        ]

        for text, base_suggestions in zip(texts, base_suggestion_batch):
            text = self._truncate_text(text, encoding)
            prompt = "Here is the text:\n" + text + "\n"

            base_labels = [
                self.project.subjects[s.subject_id].labels["en"]
                for s in base_suggestions
            ]
            prompt += "And here are the keywords:\n" + "\n".join(base_labels)
            llm_result = self._call_llm(prompt, model)
            print(llm_result)
            # try:
            #     llm_result = json.loads(llm_labels)
            # except (TypeError, json.decoder.JSONDecodeError) as err:
            #     print(err)
            #     llm_result = dict()
            results = self._map_llm_suggestions(
                llm_result,
                base_labels,
                base_suggestions,
                llm_scores_weight,
            )
            batch_results.append(results)
        return SuggestionBatch.from_sequence(batch_results, self.project.subjects)

    def _truncate_text(self, text, encoding):
        """truncate text so it contains at most MAX_PROMPT_TOKENS according to the
        OpenAI tokenizer"""

        MAX_PROMPT_TOKENS = 14000
        tokens = encoding.encode(text)
        return encoding.decode(tokens[:MAX_PROMPT_TOKENS])

    def _map_llm_suggestions(
        self,
        llm_result,
        base_labels,
        base_suggestions,
        llm_scores_weight,
    ):
        suggestions = []
        for blabel, bsuggestion in zip(base_labels, base_suggestions):
            try:
                score = llm_result[blabel]
            except KeyError:
                print(f"Base label {blabel} not found in LLM labels")
                score = bsuggestion.score  # use only base suggestion score
            subj_id = bsuggestion.subject_id

            base_scores_weight = 1.0 - llm_scores_weight
            mean_score = (
                base_scores_weight * bsuggestion.score + llm_scores_weight * score
            ) / (
                base_scores_weight + llm_scores_weight
            )  # weighted mean of LLM and base scores!
            suggestions.append(SubjectSuggestion(subject_id=subj_id, score=mean_score))
        return suggestions

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
                logprobs=True,
                # top_logprobs=2,
            )
            logprobs_completion = completion.choices[0].logprobs.content
            return self._get_results(logprobs_completion)
        except BadRequestError as err:  # openai.RateLimitError
            print(err)
            return dict()

    def _get_results(self, logprobs_completion):
        # labels, probs = [], []
        results = dict()
        line = ""
        for token in logprobs_completion:
            # print("Token:", token.token)
            # print("Linear prob:", np.round(np.exp(token.logprob) * 100, 2), "%")
            # prev_linear_prob = np.exp(token.logprob)
            prev_token = token

            line += token.token
            if "\n" in token.token:
                print("Line is: " + line)
                label, boolean_score = self._parse_line(line)
                if not label == "<failed>":
                    # results[label] = prev_linear_prob
                    results[label] = self._get_score(prev_token)
                line = ""
        return results

    def _parse_line(self, line):
        try:
            label = line.split('"')[1]
            boolean_score = line.split(":")[1].strip().replace(",", "")
        except IndexError:
            print(f"Failed parsing line: '{line}'")
            return "<failed>"
        return label, boolean_score

    def _get_score(self, token):
        linear_prob = np.exp(token.logprob)
        if token.token == "1":
            return linear_prob
        elif token.token == "0":
            return 1.0 - linear_prob
        else:
            print(token)
            return None
