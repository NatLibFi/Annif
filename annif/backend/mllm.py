"""Maui-like Lexical Matching backend"""

import collections
import math
from enum import IntEnum
from statistics import mean
import os.path
import joblib
import numpy as np
from rdflib import URIRef
from rdflib.namespace import SKOS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import annif.util
from annif.exception import NotInitializedException
from annif.suggestion import VectorSuggestionResult
from . import backend

Term = collections.namedtuple('Term', 'subject_id label is_pref')
Match = collections.namedtuple(
    'Match', 'subject_id is_pref n_tokens pos ambiguity')
Candidate = collections.namedtuple(
    'Candidate',
    'doc_length subject_id freq is_pref n_tokens ambiguity ' +
    'first_occ last_occ spread')

Feature = IntEnum(
    'Feature',
    'freq doc_freq subj_freq tfidf is_pref n_tokens ambiguity ' +
    'first_occ last_occ spread doc_length',
    start=0)


class TokenSet:
    """Represents a set of tokens (expressed as integer token IDs) that can
    be matched with another set of tokens. A TokenSet can optionally
    be associated with a subject from the vocabulary."""

    def __init__(self, tokens, subject_id=None, is_pref=False):
        self._tokens = set(tokens)
        self.subject_id = subject_id
        self.is_pref = is_pref

    def __len__(self):
        return len(self._tokens)

    def __iter__(self):
        return iter(self._tokens)

    def contains(self, other):
        """Returns True iff the tokens in the other TokenSet are all
        included within this TokenSet."""

        return other._tokens.issubset(self._tokens)

    def sample(self):
        """Return an arbitrary token from this TokenSet, or None if empty"""
        try:
            return next(iter(self._tokens))
        except StopIteration:
            return None


class TokenSetIndex:
    """A searchable index of TokenSets (representing vocabulary terms)"""

    def __init__(self):
        self._index = collections.defaultdict(set)

    def __len__(self):
        return len(self._index)

    def add(self, tset):
        """Add a TokenSet into this index"""
        token = tset.sample()
        if token is not None:
            self._index[token].add(tset)

    def search(self, tset):
        """Return the TokenSets that are contained in the given TokenSet.
        The matches are returned as a list of (TokenSet, ambiguity) pairs
        where ambiguity is an integer indicating the number of other TokenSets
        that also match the same tokens."""

        subj_tsets = {}
        subj_ambiguity = collections.Counter()

        for token in tset:
            for ts in self._index[token]:
                if not tset.contains(ts):
                    continue
                if ts.subject_id not in subj_tsets or \
                   not subj_tsets[ts.subject_id].is_pref:
                    subj_tsets[ts.subject_id] = ts

        for ts in subj_tsets.values():
            for other in subj_tsets.values():
                if ts == other:
                    continue
                if other.contains(ts):
                    subj_ambiguity.update([ts.subject_id])

        return [(ts, subj_ambiguity[ts.subject_id])
                for uri, ts in subj_tsets.items()]


class MLLMModel:
    """Maui-like Lexical Matching model"""

    def _conflate_matches(self, matches, doc_length):
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
                spread=(matches[-1].pos - matches[0].pos) / doc_length
            )
            for subject_id, matches in subj_matches.items()]

    def _generate_candidates(self, text, analyzer):
        sentences = analyzer.tokenize_sentences(text)
        sent_tokens = self._vectorizer.transform(sentences)
        matches = []

        for sent_idx, token_matrix in enumerate(sent_tokens):
            tset = TokenSet(token_matrix.nonzero()[1])
            for ts, ambiguity in self._index.search(tset):
                matches.append(Match(subject_id=ts.subject_id,
                                     is_pref=ts.is_pref,
                                     n_tokens=len(ts),
                                     pos=sent_idx,
                                     ambiguity=ambiguity))

        return self._conflate_matches(matches, len(sentences))

    def _candidates_to_features(self, candidates):
        """Convert a list of Candidates to a NumPy feature matrix"""
        matrix = np.zeros((len(candidates), len(Feature)), dtype=np.float32)
        for idx, c in enumerate(candidates):
            subj = c.subject_id
            matrix[idx, Feature.freq] = c.freq
            matrix[idx, Feature.doc_freq] = self._doc_freq[subj]
            matrix[idx, Feature.subj_freq] = self._subj_freq.get(subj, 1) - 1
            matrix[idx, Feature.tfidf] = c.freq * self._idf[subj]
            matrix[idx, Feature.is_pref] = c.is_pref
            matrix[idx, Feature.n_tokens] = c.n_tokens
            matrix[idx, Feature.ambiguity] = c.ambiguity
            matrix[idx, Feature.first_occ] = c.first_occ
            matrix[idx, Feature.last_occ] = c.last_occ
            matrix[idx, Feature.spread] = c.spread
            matrix[idx, Feature.doc_length] = c.doc_length
        return matrix

    def train(self, corpus, vocab, analyzer, params):
        graph = vocab.as_graph()
        terms = []
        subject_ids = []
        for subj_id, (uri, pref, _) in enumerate(vocab.subjects):
            if pref is None:
                continue  # deprecated subject
            subject_ids.append(subj_id)
            terms.append(Term(subject_id=subj_id, label=pref, is_pref=True))
            alts = graph.preferredLabel(URIRef(uri),
                                        lang=params['language'],
                                        labelProperties=[SKOS.altLabel])
            for label, _ in alts:
                terms.append(Term(subject_id=subj_id,
                                  label=str(label),
                                  is_pref=False))

        self._vectorizer = CountVectorizer(
            binary=True,
            tokenizer=analyzer.tokenize_words
        )
        label_corpus = self._vectorizer.fit_transform((t.label for t in terms))

        self._index = TokenSetIndex()
        for term, label_matrix in zip(terms, label_corpus):
            tokens = label_matrix.nonzero()[1]
            tset = TokenSet(tokens, term.subject_id, term.is_pref)
            self._index.add(tset)

        # frequency of subjects (by id) in the generated candidates
        self._doc_freq = collections.Counter()
        # frequency of manually assigned subjects ("domain keyphraseness")
        self._subj_freq = collections.Counter()
        doc_count = 0
        train_X = []
        train_y = []
        for idx, doc in enumerate(corpus.documents):
            doc_subject_ids = [vocab.subjects.by_uri(uri)
                               for uri in doc.uris]
            self._subj_freq.update(doc_subject_ids)
            candidates = self._generate_candidates(doc.text, analyzer)
            self._doc_freq.update([c.subject_id for c in candidates])
            train_X += candidates
            train_y += [(c.subject_id in doc_subject_ids) for c in candidates]
            doc_count += 1

        # precalculate idf values for candidate subjects
        self._idf = collections.defaultdict(float)
        for subj_id in subject_ids:
            self._idf[uri] = math.log((doc_count + 1) /
                                      (self._doc_freq[subj_id] + 1)) + 1

        # define a sklearn pipeline with transformer and classifier
        # TODO: make hyperparameters configurable
        self._model = Pipeline(
            steps=[
                ('transformer', FunctionTransformer(
                    self._candidates_to_features)),
                ('classifier', BaggingClassifier(
                    DecisionTreeClassifier(
                        min_samples_leaf=int(params['min_samples_leaf']),
                        max_leaf_nodes=int(params['max_leaf_nodes'])
                    ), max_samples=float(params['max_samples'])))])
        # fit the model on the training corpus
        self._model.fit(train_X, train_y)

    def predict(self, text, analyzer):
        candidates = self._generate_candidates(text, analyzer)
        if not candidates:
            return []
        scores = self._model.predict_proba(candidates)
        subj_scores = [(score[1], c.subject_id)
                       for score, c in zip(scores, candidates)]
        return sorted(subj_scores, reverse=True)


class MLLMBackend(backend.AnnifBackend):
    """Maui-like Lexical Matching backend for Annif"""
    name = "mllm"
    needs_subject_index = True

    # defaults for unitialized instances
    _model = None

    MODEL_FILE = 'model'

    DEFAULT_PARAMETERS = {
        'min_samples_leaf': 20,
        'max_leaf_nodes': 1000,
        'max_samples': 0.9
    }

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def initialize(self):
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            self.debug('loading model from {}'.format(path))
            if os.path.exists(path):
                self._model = joblib.load(path)
            else:
                raise NotInitializedException(
                    'model {} not found'.format(path),
                    backend_id=self.backend_id)

    def _train(self, corpus, params):
        # TODO: check for "cached" corpus
        self.info('starting train')
        self._model = MLLMModel()
        self._model.train(
            corpus,
            self.project.vocab,
            self.project.analyzer,
            params)
        self.info('saving model')
        annif.util.atomic_save(
            self._model,
            self.datadir,
            self.MODEL_FILE,
            method=joblib.dump)

    def _suggest(self, text, params):
        vector = np.zeros(len(self.project.subjects), dtype=np.float32)
        for score, subject_id in self._model.predict(text,
                                                     self.project.analyzer):
            vector[subject_id] = score
        result = VectorSuggestionResult(vector)
        return result.filter(self.project.subjects,
                             limit=int(params['limit']))
