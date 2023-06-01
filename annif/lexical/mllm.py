"""MLLM (Maui-like Lexical Matchin) model for Annif"""
from __future__ import annotations

import collections
import math
from enum import IntEnum
from statistics import mean
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from rdflib.namespace import SKOS
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

import annif.parallel
import annif.util
from annif.exception import OperationFailedException
from annif.lexical.tokenset import TokenSet, TokenSetIndex
from annif.lexical.util import (
    get_subject_labels,
    make_collection_matrix,
    make_relation_matrix,
)

if TYPE_CHECKING:
    from collections import defaultdict

    from rdflib.graph import Graph
    from rdflib.term import URIRef

    from annif.analyzer import Analyzer
    from annif.corpus.document import DocumentCorpus
    from annif.vocab import AnnifVocabulary

Term = collections.namedtuple("Term", "subject_id label is_pref")

Match = collections.namedtuple("Match", "subject_id is_pref n_tokens pos ambiguity")

Candidate = collections.namedtuple(
    "Candidate",
    "doc_length subject_id freq is_pref n_tokens ambiguity "
    + "first_occ last_occ spread",
)

ModelData = collections.namedtuple(
    "ModelData", "broader narrower related collection " + "doc_freq subj_freq idf"
)

Feature = IntEnum(
    "Feature",
    "freq doc_freq subj_freq tfidf is_pref n_tokens ambiguity "
    + "first_occ last_occ spread doc_length "
    + "broader narrower related collection",
    start=0,
)


def conflate_matches(matches: list[Match], doc_length: int) -> list[Candidate]:
    subj_matches = collections.defaultdict(list)
    for match in matches:
        subj_matches[match.subject_id].append(match)
    return [
        Candidate(
            doc_length=doc_length,
            subject_id=subject_id,
            freq=len(matches) / doc_length,
            is_pref=mean((float(m.is_pref) for m in matches)),
            n_tokens=mean((m.n_tokens for m in matches)),
            ambiguity=mean((m.ambiguity for m in matches)),
            first_occ=matches[0].pos / doc_length,
            last_occ=matches[-1].pos / doc_length,
            spread=(matches[-1].pos - matches[0].pos) / doc_length,
        )
        for subject_id, matches in subj_matches.items()
    ]


def generate_candidates(
    text: str,
    analyzer: Analyzer,
    vectorizer: CountVectorizer,
    index: TokenSetIndex,
) -> list[Candidate]:
    sentences = analyzer.tokenize_sentences(text)
    sent_tokens = vectorizer.transform(sentences)
    matches = []

    for sent_idx, token_matrix in enumerate(sent_tokens):
        tset = TokenSet(token_matrix.nonzero()[1])
        for ts, ambiguity in index.search(tset):
            matches.append(
                Match(
                    subject_id=ts.subject_id,
                    is_pref=ts.is_pref,
                    n_tokens=len(ts),
                    pos=sent_idx,
                    ambiguity=ambiguity,
                )
            )

    return conflate_matches(matches, len(sentences))


def candidates_to_features(
    candidates: list[Candidate], mdata: "ModelData"
) -> np.ndarray:
    """Convert a list of Candidates to a NumPy feature matrix"""

    matrix = np.zeros((len(candidates), len(Feature)), dtype=np.float32)
    c_ids = [c.subject_id for c in candidates]
    c_vec = np.zeros(mdata.related.shape[0], dtype=bool)
    c_vec[c_ids] = True
    broader = mdata.broader.multiply(c_vec).sum(axis=1)
    narrower = mdata.narrower.multiply(c_vec).sum(axis=1)
    related = mdata.related.multiply(c_vec).sum(axis=1)
    collection = mdata.collection.multiply(c_vec).T.dot(mdata.collection).sum(axis=0)
    for idx, c in enumerate(candidates):
        subj = c.subject_id
        matrix[idx, Feature.freq] = c.freq
        matrix[idx, Feature.doc_freq] = mdata.doc_freq[subj]
        matrix[idx, Feature.subj_freq] = mdata.subj_freq.get(subj, 1) - 1
        matrix[idx, Feature.tfidf] = c.freq * mdata.idf[subj]
        matrix[idx, Feature.is_pref] = c.is_pref
        matrix[idx, Feature.n_tokens] = c.n_tokens
        matrix[idx, Feature.ambiguity] = c.ambiguity
        matrix[idx, Feature.first_occ] = c.first_occ
        matrix[idx, Feature.last_occ] = c.last_occ
        matrix[idx, Feature.spread] = c.spread
        matrix[idx, Feature.doc_length] = c.doc_length
        matrix[idx, Feature.broader] = broader[subj, 0] / len(c_ids)
        matrix[idx, Feature.narrower] = narrower[subj, 0] / len(c_ids)
        matrix[idx, Feature.related] = related[subj, 0] / len(c_ids)
        matrix[idx, Feature.collection] = collection[0, subj] / len(c_ids)
    return matrix


class MLLMCandidateGenerator(annif.parallel.BaseWorker):
    @classmethod
    def generate_candidates(cls, doc_subject_set, text):
        candidates = generate_candidates(text, **cls.args)  # pragma: no cover
        return doc_subject_set, candidates  # pragma: no cover


class MLLMFeatureConverter(annif.parallel.BaseWorker):
    @classmethod
    def candidates_to_features(cls, candidates):
        return candidates_to_features(candidates, **cls.args)  # pragma: no cover


class MLLMModel:
    """Maui-like Lexical Matching model"""

    def generate_candidates(self, text: str, analyzer: Analyzer) -> list[Candidate]:
        return generate_candidates(text, analyzer, self._vectorizer, self._index)

    @property
    def _model_data(self) -> ModelData:
        return ModelData(
            broader=self._broader_matrix,
            narrower=self._narrower_matrix,
            related=self._related_matrix,
            collection=self._collection_matrix,
            doc_freq=self._doc_freq,
            subj_freq=self._subj_freq,
            idf=self._idf,
        )

    def _candidates_to_features(self, candidates: list[Candidate]) -> np.ndarray:
        return candidates_to_features(candidates, self._model_data)

    @staticmethod
    def _get_label_props(params: dict[str, Any]) -> tuple[list[URIRef], list[URIRef]]:
        pref_label_props = [SKOS.prefLabel]

        if annif.util.boolean(params["use_hidden_labels"]):
            nonpref_label_props = [SKOS.altLabel, SKOS.hiddenLabel]
        else:
            nonpref_label_props = [SKOS.altLabel]

        return (pref_label_props, nonpref_label_props)

    def _prepare_terms(
        self,
        graph: Graph,
        vocab: AnnifVocabulary,
        params: dict[str, Any],
    ) -> tuple[list[Term], list[int]]:
        pref_label_props, nonpref_label_props = self._get_label_props(params)

        terms = []
        subject_ids = []
        for subj_id, subject in vocab.subjects.active:
            subject_ids.append(subj_id)

            for label in get_subject_labels(
                graph, subject.uri, pref_label_props, params["language"]
            ):
                terms.append(Term(subject_id=subj_id, label=label, is_pref=True))

            for label in get_subject_labels(
                graph, subject.uri, nonpref_label_props, params["language"]
            ):
                terms.append(Term(subject_id=subj_id, label=label, is_pref=False))

        return (terms, subject_ids)

    def _prepare_relations(self, graph: Graph, vocab: AnnifVocabulary) -> None:
        self._broader_matrix = make_relation_matrix(graph, vocab, SKOS.broader)
        self._narrower_matrix = make_relation_matrix(graph, vocab, SKOS.narrower)
        self._related_matrix = make_relation_matrix(graph, vocab, SKOS.related)
        self._collection_matrix = make_collection_matrix(graph, vocab)

    def _prepare_train_index(
        self,
        vocab: AnnifVocabulary,
        analyzer: Analyzer,
        params: dict[str, Any],
    ) -> list[int]:
        graph = vocab.as_graph()
        terms, subject_ids = self._prepare_terms(graph, vocab, params)
        self._prepare_relations(graph, vocab)

        self._vectorizer = CountVectorizer(
            binary=True, tokenizer=analyzer.tokenize_words
        )
        label_corpus = self._vectorizer.fit_transform((t.label for t in terms))

        # frequency of each token used in labels - how rare each word is
        token_freq = np.bincount(label_corpus.indices, minlength=label_corpus.shape[1])

        self._index = TokenSetIndex()
        for term, label_matrix in zip(terms, label_corpus):
            tokens = label_matrix.nonzero()[1]
            # sort tokens by frequency - use the rarest token as index key
            tokens = sorted(tokens, key=token_freq.__getitem__)
            tset = TokenSet(tokens, term.subject_id, term.is_pref)
            self._index.add(tset)

        return subject_ids

    def _prepare_train_data(
        self, corpus: DocumentCorpus, analyzer: Analyzer, n_jobs: int
    ) -> tuple[list[list[Candidate]], list[bool]]:
        # frequency of subjects (by id) in the generated candidates
        self._doc_freq = collections.Counter()
        # frequency of manually assigned subjects ("domain keyphraseness")
        self._subj_freq = collections.Counter()
        train_x = []
        train_y = []

        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        cg_args = {
            "analyzer": analyzer,
            "vectorizer": self._vectorizer,
            "index": self._index,
        }

        with pool_class(
            jobs, initializer=MLLMCandidateGenerator.init, initargs=(cg_args,)
        ) as pool:
            params = ((doc.subject_set, doc.text) for doc in corpus.documents)
            for doc_subject_ids, candidates in pool.starmap(
                MLLMCandidateGenerator.generate_candidates, params, 10
            ):
                self._subj_freq.update(doc_subject_ids)
                self._doc_freq.update([c.subject_id for c in candidates])
                train_x.append(candidates)
                train_y += [(c.subject_id in doc_subject_ids) for c in candidates]

        return (train_x, train_y)

    def _calculate_idf(
        self, subject_ids: list[int], doc_count: int
    ) -> defaultdict[int, float]:
        idf = collections.defaultdict(float)
        for subj_id in subject_ids:
            idf[subj_id] = math.log((doc_count + 1) / (self._doc_freq[subj_id] + 1)) + 1

        return idf

    def _prepare_features(
        self, train_x: list[list[Candidate]], n_jobs: int
    ) -> list[np.ndarray]:
        fc_args = {"mdata": self._model_data}
        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        with pool_class(
            jobs, initializer=MLLMFeatureConverter.init, initargs=(fc_args,)
        ) as pool:
            features = pool.map(
                MLLMFeatureConverter.candidates_to_features, train_x, 10
            )

        return features

    def prepare_train(
        self,
        corpus: DocumentCorpus,
        vocab: AnnifVocabulary,
        analyzer: Analyzer,
        params: dict[str, Any],
        n_jobs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # create an index from the vocabulary terms
        subject_ids = self._prepare_train_index(vocab, analyzer, params)

        # convert the corpus into train data
        train_x, train_y = self._prepare_train_data(corpus, analyzer, n_jobs)

        # precalculate idf values for all candidate subjects
        self._idf = self._calculate_idf(subject_ids, len(train_x))

        # convert the train data into feature values
        features = self._prepare_features(train_x, n_jobs)

        return (np.vstack(features), np.array(train_y))

    def _create_classifier(self, params: dict[str, Any]) -> BaggingClassifier:
        return BaggingClassifier(
            DecisionTreeClassifier(
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_leaf_nodes=int(params["max_leaf_nodes"]),
            ),
            max_samples=float(params["max_samples"]),
        )

    def train(
        self,
        train_x: np.ndarray | list[tuple[int, int]],
        train_y: list[bool] | np.ndarray,
        params: dict[str, Any],
    ) -> None:
        # fit the model on the training corpus
        self._classifier = self._create_classifier(params)
        self._classifier.fit(train_x, train_y)
        # sanity check: verify that the classifier has seen both classes
        if self._classifier.n_classes_ != 2:
            raise OperationFailedException(
                "Unable to create classifier: "
                + "Not enough positive and negative examples "
                + "in the training data. Please check that your training "
                + "data matches your vocabulary."
            )

    def _prediction_to_list(
        self, scores: np.ndarray, candidates: list[Candidate]
    ) -> list[tuple[np.float64, int]]:
        subj_scores = [(score[1], c.subject_id) for score, c in zip(scores, candidates)]
        return sorted(subj_scores, reverse=True)

    def predict(self, candidates: list[Candidate]) -> list[tuple[np.float64, int]]:
        if not candidates:
            return []
        features = self._candidates_to_features(candidates)
        scores = self._classifier.predict_proba(features)
        return self._prediction_to_list(scores, candidates)

    def save(self, filename: str) -> list[str]:
        return joblib.dump(self, filename)

    @staticmethod
    def load(filename: str) -> MLLMModel:
        return joblib.load(filename)
