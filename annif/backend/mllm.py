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
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import annif.util
from annif.exception import NotInitializedException
from annif.suggestion import VectorSuggestionResult
from . import backend
from . import hyperopt

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
    'first_occ last_occ spread doc_length ' +
    'related',
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

    def generate_candidates(self, text, analyzer):
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
        c_ids = [c.subject_id for c in candidates]
        c_vec = np.zeros(self._related_matrix.shape[0], dtype=np.bool)
        c_vec[c_ids] = True
        rels = self._related_matrix.multiply(c_vec).sum(axis=1)
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
            matrix[idx, Feature.related] = rels[subj, 0] / len(c_ids)
        return matrix

    def _prepare_terms(self, graph, vocab, params):
        terms = []
        subject_ids = []
        for subj_id, (uri, pref, _) in enumerate(vocab.subjects):
            if pref is None:
                continue  # deprecated subject
            subject_ids.append(subj_id)
            terms.append(Term(subject_id=subj_id, label=pref, is_pref=True))

            if annif.util.boolean(params['use_hidden_labels']):
                label_props = [SKOS.altLabel, SKOS.hiddenLabel]
            else:
                label_props = [SKOS.altLabel]

            for prop in label_props:
                for label in graph.objects(URIRef(uri), prop):
                    if label.language != params['language']:
                        continue
                    terms.append(Term(subject_id=subj_id,
                                      label=str(label),
                                      is_pref=False))
        return (terms, subject_ids)

    def _prepare_relations(self, graph, vocab):
        n_subj = len(vocab.subjects)
        self._related_matrix = lil_matrix((n_subj, n_subj), dtype=np.bool)

        for subj_id, (uri, pref, _) in enumerate(vocab.subjects):
            if pref is None:
                continue  # deprecated subject
            for related in graph.objects(URIRef(uri), SKOS.related):
                broad_id = vocab.subjects.by_uri(str(related), warnings=False)
                if broad_id is not None:
                    self._related_matrix[subj_id, broad_id] = True

    def prepare_train(self, corpus, vocab, analyzer, params):
        graph = vocab.as_graph()
        terms, subject_ids = self._prepare_terms(graph, vocab, params)
        self._prepare_relations(graph, vocab)

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
        train_x = []
        train_y = []
        for idx, doc in enumerate(corpus.documents):
            doc_subject_ids = [vocab.subjects.by_uri(uri)
                               for uri in doc.uris]
            self._subj_freq.update(doc_subject_ids)
            candidates = self.generate_candidates(doc.text, analyzer)
            self._doc_freq.update([c.subject_id for c in candidates])
            train_x.append(candidates)
            train_y += [(c.subject_id in doc_subject_ids) for c in candidates]
            doc_count += 1

        # precalculate idf values for candidate subjects
        self._idf = collections.defaultdict(float)
        for subj_id in subject_ids:
            self._idf[subj_id] = math.log((doc_count + 1) /
                                          (self._doc_freq[subj_id] + 1)) + 1
        return (np.vstack([self._candidates_to_features(candidates)
                           for candidates in train_x]), np.array(train_y))

    def _create_classifier(self, params):
        return BaggingClassifier(
            DecisionTreeClassifier(
                min_samples_leaf=int(params['min_samples_leaf']),
                max_leaf_nodes=int(params['max_leaf_nodes'])
            ), max_samples=float(params['max_samples']))

    def train(self, train_x, train_y, params):
        # fit the model on the training corpus
        self._classifier = self._create_classifier(params)
        self._classifier.fit(train_x, train_y)

    def _prediction_to_list(self, scores, candidates):
        subj_scores = [(score[1], c.subject_id)
                       for score, c in zip(scores, candidates)]
        return sorted(subj_scores, reverse=True)

    def predict(self, candidates):
        if not candidates:
            return []
        features = self._candidates_to_features(candidates)
        scores = self._classifier.predict_proba(features)
        return self._prediction_to_list(scores, candidates)


class MLLMOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the MLLM backend"""

    def _prepare(self, n_jobs=1):
        self._backend.initialize()
        self._train_x, self._train_y = self._backend._load_train_data()
        self._candidates = []
        self._gold_subjects = []

        # TODO parallelize generation of candidates
        for doc in self._corpus.documents:
            candidates = self._backend._generate_candidates(doc.text)
            self._candidates.append(candidates)
            self._gold_subjects.append(
                annif.corpus.SubjectSet((doc.uris, doc.labels)))

    def _objective(self, trial):
        params = {
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 100, 2000),
            'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
            'use_hidden_labels':
                trial.suggest_categorical('use_hidden_labels', [True, False]),
            'limit': 100
        }
        model = self._backend._model._create_classifier(params)
        model.fit(self._train_x, self._train_y)

        batch = annif.eval.EvaluationBatch(self._backend.project.subjects)
        for goldsubj, candidates in zip(self._gold_subjects, self._candidates):
            if candidates:
                features = \
                    self._backend._model._candidates_to_features(candidates)
                scores = model.predict_proba(features)
                ranking = self._backend._model._prediction_to_list(
                    scores, candidates)
            else:
                ranking = []
            results = self._backend._prediction_to_result(ranking, params)
            batch.evaluate(results, goldsubj)
        results = batch.results(metrics=[self._metric])
        return results[self._metric]

    def _postprocess(self, study):
        bp = study.best_params
        lines = [
            f"min_samples_leaf={bp['min_samples_leaf']}",
            f"max_leaf_nodes={bp['max_leaf_nodes']}",
            f"max_samples={bp['max_samples']:.4f}",
            f"use_hidden_labels={bp['use_hidden_labels']}"
        ]
        return hyperopt.HPRecommendation(lines=lines, score=study.best_value)


class MLLMBackend(hyperopt.AnnifHyperoptBackend):
    """Maui-like Lexical Matching backend for Annif"""
    name = "mllm"
    needs_subject_index = True

    # defaults for unitialized instances
    _model = None

    MODEL_FILE = 'mllm-model.gz'
    TRAIN_FILE = 'mllm-train.gz'

    DEFAULT_PARAMETERS = {
        'min_samples_leaf': 20,
        'max_leaf_nodes': 1000,
        'max_samples': 0.9,
        'use_hidden_labels': False
    }

    def get_hp_optimizer(self, corpus, metric):
        return MLLMOptimizer(self, corpus, metric)

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def _load_model(self):
        path = os.path.join(self.datadir, self.MODEL_FILE)
        self.debug('loading model from {}'.format(path))
        if os.path.exists(path):
            return joblib.load(path)
        else:
            raise NotInitializedException(
                'model {} not found'.format(path),
                backend_id=self.backend_id)

    def _load_train_data(self):
        path = os.path.join(self.datadir, self.TRAIN_FILE)
        if os.path.exists(path):
            return joblib.load(path)
        else:
            raise NotInitializedException(
                'train data file {} not found'.format(path),
                backend_id=self.backend_id)

    def initialize(self):
        if self._model is None:
            self._model = self._load_model()

    def _train(self, corpus, params):
        self.info('starting train')
        if corpus != 'cached':
            self.info("preparing training data")
            self._model = MLLMModel()
            train_data = self._model.prepare_train(corpus,
                                                   self.project.vocab,
                                                   self.project.analyzer,
                                                   params)
            annif.util.atomic_save(train_data,
                                   self.datadir,
                                   self.TRAIN_FILE,
                                   method=joblib.dump)
        else:
            self.info("reusing cached training data from previous run")
            self._model = self._load_model()
            train_data = self._load_train_data()

        self.info("training model")
        self._model.train(train_data[0], train_data[1], params)

        self.info('saving model')
        annif.util.atomic_save(
            self._model,
            self.datadir,
            self.MODEL_FILE,
            method=joblib.dump)

    def _generate_candidates(self, text):
        return self._model.generate_candidates(text, self.project.analyzer)

    def _prediction_to_result(self, prediction, params):
        vector = np.zeros(len(self.project.subjects), dtype=np.float32)
        for score, subject_id in prediction:
            vector[subject_id] = score
        result = VectorSuggestionResult(vector)
        return result.filter(self.project.subjects,
                             limit=int(params['limit']))

    def _suggest(self, text, params):
        candidates = self._generate_candidates(text)
        prediction = self._model.predict(candidates)
        return self._prediction_to_result(prediction, params)
