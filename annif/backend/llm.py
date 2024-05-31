"""TODO"""

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
            answer, probabilities = self._call_llm(prompt, model)
            print(answer)
            print(probabilities)
            try:
                llm_result = json.loads(answer)
            except (TypeError, json.decoder.JSONDecodeError) as err:
                print(err)
                llm_result = dict()
            results = self._get_llm_suggestions(
                llm_result,
                base_labels,
                base_suggestions,
                llm_scores_weight,
                # probabilities,
                # llm_probs_weight,
            )
            batch_results.append(results)
        return SuggestionBatch.from_sequence(batch_results, self.project.subjects)

    def _truncate_text(self, text, encoding):
        """truncate text so it contains at most MAX_PROMPT_TOKENS according to the
        OpenAI tokenizer"""

        MAX_PROMPT_TOKENS = 14000
        tokens = encoding.encode(text)
        return encoding.decode(tokens[:MAX_PROMPT_TOKENS])

    def _get_llm_suggestions(
        self,
        llm_result,
        base_labels,
        base_suggestions,
        llm_scores_weight,
        # probabilities,
        # llm_probs_weight,
    ):
        suggestions = []
        # print(f"LLM result: {llm_result}")
        for blabel, bsuggestion in zip(base_labels, base_suggestions):
            # score = llm_result.get(blabel, 0)
            try:
                score = llm_result[blabel]
                # probability = probabilities[blabel]
            except KeyError:
                print(f"Base label {blabel} not found in LLM labels")
                score = bsuggestion.score  # use only base suggestion score
                # probability = 0.0
            subj_id = bsuggestion.subject_id

            base_scores_weight = 1.0 - llm_scores_weight
            mean_score = (
                base_scores_weight * bsuggestion.score
                + llm_scores_weight * score  # * probability * llm_probs_weight
            ) / (
                base_scores_weight + llm_scores_weight  # * probability * llm_probs_weight
            )  # weighted mean of LLM and base scores!
            suggestions.append(SubjectSuggestion(subject_id=subj_id, score=mean_score))
        return suggestions

    # async def _call_llm(self, prompt: str, model: str):
    def _call_llm(self, prompt: str, model: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            # completion = await client.chat.completions.create(
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
                # logprobs=True,
            )
            # return completion.choices[0].message.content

            answer = completion.choices[0].message.content
            # lines = self._get_logprobs(completion.choices[0].logprobs.content)
            # probs = self._get_probs(lines)
            # return answer, probs
            return answer, dict()
        except BadRequestError as err:  # openai.RateLimitError
            print(err)
            return "{}", dict()

    def _get_logprobs(self, content):
        import numpy as np

        lines = []
        joint_logprob = 0.0
        line = ""
        line_joint_logprob = 0.0
        for token in content:
            # print("Token:", token.token)
            # print("Log prob:", token.logprob)
            # print("Linear prob:", np.round(np.exp(token.logprob) * 100, 2), "%")
            # print("Bytes:", token.bytes, "\n")
            # aggregated_bytes += token.bytes
            joint_logprob += token.logprob

            line += token.token
            line_joint_logprob += token.logprob
            if "\n" in token.token:
                # print("Line is: "+ line)
                line_prob = np.exp(line_joint_logprob)
                # print("Line's linear prob:",  np.round(line_prob * 100, 2), "%")

                lines.append((line, line_prob))
                line = ""
                line_joint_logprob = 0.0
        #         print()
        # print()
        # print("Joint log prob:", joint_logprob)
        # print("Joint prob:", np.round(np.exp(joint_logprob) * 100, 2), "%")
        return lines

    # def _get_probs(self, lines):
    #     probs = dict()
    #     for line, prob in lines:
    #         try:
    #             label = line.split('"')[1]
    #         except IndexError:
    #             print("Failed parsing line: " + line)
    #             continue  # Not a line with label
    #         # probs[label] = 1.0
    #         probs[label] = prob
        return probs
