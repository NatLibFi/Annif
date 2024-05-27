"""TODO"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import joblib
import tiktoken
from openai import AzureOpenAI, BadRequestError
from rdflib.namespace import SKOS

import annif.eval
import annif.parallel
import annif.util
from annif.exception import ConfigurationException, NotSupportedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import backend

# from openai import AsyncAzureOpenAI


if TYPE_CHECKING:
    from datetime import datetime

    from rdflib.term import URIRef

    from annif.corpus.document import DocumentCorpus


class BaseLLMBackend(backend.AnnifBackend):
    # """Base class for TODO backends"""

    def initialize(self, parallel: bool = False) -> None:
        # initialize all the source projects
        params = self._get_backend_params(None)

        # self.client = AsyncAzureOpenAI(
        self.client = AzureOpenAI(
            azure_endpoint=params["endpoint"],
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
        )
        self._initialize_index()


class LLMBackend(BaseLLMBackend, backend.AnnifBackend):
    # """TODO backend that combines results from multiple projects"""

    name = "llm"
    # defaults for uninitialized instances
    _index = None
    INDEX_FILE = "llm-index"

    DEFAULT_PARAMETERS = {
        "label_types": ["prefLabel", "altLabel"],
        "remove_parentheses": False,
    }

    system_prompt = """
        You are a professional subject indexer.
        You will be given a text. Your task is to give a list of keywords to describe
        the text along scores for the keywords with a value between 0.0 and 1.0. The
        score value should depend on how well the keyword represents the text: a perfect
        keyword should have score 1.0 and completely unrelated keyword score
        0.0. You must output JSON with keywords as field names and add their scores
        as field values.
    """
    # Give zero or very low score to the keywords that do not describe the text.

    @property
    def is_trained(self) -> bool:
        True

    @property
    def modification_time(self) -> datetime | None:
        None

    def _train(self, corpus: DocumentCorpus, params: dict[str, Any], jobs: int = 0):
        raise NotSupportedException("Training LLM backend is not possible.")

    @property
    def label_types(self) -> list[URIRef]:
        if isinstance(self.params["label_types"], str):  # Label types set by user
            label_types = [lt.strip() for lt in self.params["label_types"].split(",")]
            self._validate_label_types(label_types)
        else:
            label_types = self.params["label_types"]  # The defaults
        return [getattr(SKOS, lt) for lt in label_types]

    def _validate_label_types(self, label_types: list[str]) -> None:
        for lt in label_types:
            if lt not in ("prefLabel", "altLabel", "hiddenLabel"):
                raise ConfigurationException(
                    f"invalid label type {lt}", backend_id=self.backend_id
                )

    def _initialize_index(self) -> None:
        if self._index is None:
            path = os.path.join(self.datadir, self.INDEX_FILE)
            if os.path.exists(path):
                self._index = joblib.load(path)
                self.debug(f"Loaded index from {path} with {len(self._index)} labels")
            else:
                self.info("Creating index")
                self._index = self._create_index()
                self._save_index(path)
                self.info(f"Created index with {len(self._index)} labels")

    def _save_index(self, path: str) -> None:
        annif.util.atomic_save(
            self._index, self.datadir, self.INDEX_FILE, method=joblib.dump
        )

    def _create_index(self) -> dict[str, set[str]]:
        index = defaultdict(set)
        skos_vocab = self.project.vocab.skos
        for concept in skos_vocab.concepts:
            uri = str(concept)
            labels_by_lang = skos_vocab.get_concept_labels(concept, self.label_types)
            for label in labels_by_lang[self.params["language"]]:
                # label = self._normalize_label(label)
                index[label].add(uri)
        index.pop("", None)  # Remove possible empty string entry
        return dict(index)

    def _suggest(self, text: str, params: dict[str, Any]) -> SuggestionBatch:
        model = params["model"]
        limit = int(params["limit"])

        encoding = tiktoken.encoding_for_model(model.rsplit("-", 1)[0])

        text = self._truncate_text(text, encoding)
        prompt = "Here is the text:\n" + text + "\n"

        answer = self._call_llm(prompt, model)
        try:
            llm_result = json.loads(answer)
        except (TypeError, json.decoder.JSONDecodeError) as err:
            print(err)
            llm_result = dict()

        keyphrases = [(kp, score) for kp, score in llm_result.items()]
        suggestions = self._keyphrases2suggestions(keyphrases)

        subject_suggestions = [
            SubjectSuggestion(subject_id=self.project.subjects.by_uri(uri), score=score)
            for uri, score in suggestions[:limit]
            if score > 0.0
        ]
        return subject_suggestions

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
        # print(prompt) #[-10000:])
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

    def _keyphrases2suggestions(
        self, keyphrases: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        suggestions = []
        not_matched = []
        for kp, score in keyphrases:
            uris = self._keyphrase2uris(kp)
            for uri in uris:
                suggestions.append((uri, score))
            if not uris:
                not_matched.append((kp, score))
        # Remove duplicate uris, conflating the scores
        suggestions = self._combine_suggestions(suggestions)
        self.debug(
            "Keyphrases not matched:\n"
            + "\t".join(
                [
                    kp[0] + " " + str(kp[1])
                    for kp in sorted(not_matched, reverse=True, key=lambda kp: kp[1])
                ]
            )
        )
        return suggestions

    def _keyphrase2uris(self, keyphrase: str) -> set[str]:
        keyphrase = self._normalize_phrase(keyphrase)
        keyphrase = self._sort_phrase(keyphrase)
        return self._index.get(keyphrase, [])

    def _normalize_label(self, label: str) -> str:
        label = str(label)
        if annif.util.boolean(self.params["remove_parentheses"]):
            label = re.sub(r" \(.*\)", "", label)
        normalized_label = self._normalize_phrase(label)
        return self._sort_phrase(normalized_label)

    def _normalize_phrase(self, phrase: str) -> str:
        return " ".join(self.project.analyzer.tokenize_words(phrase, filter=False))

    def _sort_phrase(self, phrase: str) -> str:
        words = phrase.split()
        return " ".join(sorted(words))

    def _combine_suggestions(
        self, suggestions: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        combined_suggestions = {}
        for uri, score in suggestions:
            if uri not in combined_suggestions:
                combined_suggestions[uri] = score
            else:
                old_score = combined_suggestions[uri]
                combined_suggestions[uri] = self._combine_scores(score, old_score)
        return list(combined_suggestions.items())

    def _combine_scores(self, score1: float, score2: float) -> float:
        # The result is never smaller than the greater input
        score1 = score1 / 2 + 0.5
        score2 = score2 / 2 + 0.5
        confl = score1 * score2 / (score1 * score2 + (1 - score1) * (1 - score2))
        return (confl - 0.5) * 2
