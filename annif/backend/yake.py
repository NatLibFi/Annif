"""Annif backend using Yake keyword extraction"""
# For license remarks of this backend see README.md:
# https://github.com/NatLibFi/Annif#license.

import yake
import joblib
import os.path
import re
from collections import defaultdict
from rdflib.namespace import SKOS
import annif.util
from . import backend
from annif.suggestion import SubjectSuggestion, ListSuggestionResult
from annif.exception import ConfigurationException, NotSupportedException


class YakeBackend(backend.AnnifBackend):
    """Yake based backend for Annif"""
    name = "yake"
    needs_subject_index = False

    # defaults for uninitialized instances
    _index = None
    _graph = None
    INDEX_FILE = 'yake-index'

    DEFAULT_PARAMETERS = {
        'max_ngram_size': 4,
        'deduplication_threshold': 0.9,
        'deduplication_algo': 'levs',
        'window_size': 1,
        'num_keywords': 100,
        'features': None,
        'label_types': ['prefLabel', 'altLabel'],
        'remove_parentheses': False
    }

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)
        return params

    @property
    def is_trained(self):
        return True

    @property
    def label_types(self):
        if type(self.params['label_types']) == str:  # Label types set by user
            label_types = [lt.strip() for lt
                           in self.params['label_types'].split(',')]
            self._validate_label_types(label_types)
        else:
            label_types = self.params['label_types']  # The defaults
        return [getattr(SKOS, lt) for lt in label_types]

    def _validate_label_types(self, label_types):
        for lt in label_types:
            if lt not in ('prefLabel', 'altLabel', 'hiddenLabel'):
                raise ConfigurationException(
                    f'invalid label type {lt}', backend_id=self.backend_id)

    def initialize(self, parallel=False):
        self._initialize_index()

    def _initialize_index(self):
        if self._index is None:
            path = os.path.join(self.datadir, self.INDEX_FILE)
            if os.path.exists(path):
                self._index = joblib.load(path)
                self.debug(
                    f'Loaded index from {path} with {len(self._index)} labels')
            else:
                self.info('Creating index')
                self._index = self._create_index()
                self._save_index(path)
                self.info(f'Created index with {len(self._index)} labels')

    def _save_index(self, path):
        annif.util.atomic_save(
            self._index,
            self.datadir,
            self.INDEX_FILE,
            method=joblib.dump)

    def _create_index(self):
        index = defaultdict(set)
        skos_vocab = self.project.vocab.skos
        for concept in skos_vocab.concepts:
            uri = str(concept)
            labels = skos_vocab.get_concept_labels(
                concept, self.label_types, self.params['language'])
            for label in labels:
                label = self._normalize_label(label)
                index[label].add(uri)
        index.pop('', None)  # Remove possible empty string entry
        return dict(index)

    def _normalize_label(self, label):
        label = str(label)
        if annif.util.boolean(self.params['remove_parentheses']):
            label = re.sub(r' \(.*\)', '', label)
        normalized_label = self._normalize_phrase(label)
        return self._sort_phrase(normalized_label)

    def _normalize_phrase(self, phrase):
        return ' '.join(self.project.analyzer.tokenize_words(phrase,
                                                             filter=False))

    def _sort_phrase(self, phrase):
        words = phrase.split()
        return ' '.join(sorted(words))

    def _suggest(self, text, params):
        self.debug(
            f'Suggesting subjects for text "{text[:20]}..." (len={len(text)})')
        limit = int(params['limit'])

        self._kw_extractor = yake.KeywordExtractor(
            lan=params['language'],
            n=int(params['max_ngram_size']),
            dedupLim=float(params['deduplication_threshold']),
            dedupFunc=params['deduplication_algo'],
            windowsSize=int(params['window_size']),
            top=int(params['num_keywords']),
            features=self.params['features'])
        keyphrases = self._kw_extractor.extract_keywords(text)
        suggestions = self._keyphrases2suggestions(keyphrases)

        subject_suggestions = [SubjectSuggestion(
                uri=uri,
                label=None,
                notation=None,
                score=score)
                for uri, score in suggestions[:limit] if score > 0.0]
        return ListSuggestionResult.create_from_index(subject_suggestions,
                                                      self.project.subjects)

    def _keyphrases2suggestions(self, keyphrases):
        suggestions = []
        not_matched = []
        for kp, score in keyphrases:
            uris = self._keyphrase2uris(kp)
            for uri in uris:
                suggestions.append(
                    (uri, self._transform_score(score)))
            if not uris:
                not_matched.append((kp, self._transform_score(score)))
        # Remove duplicate uris, conflating the scores
        suggestions = self._combine_suggestions(suggestions)
        self.debug('Keyphrases not matched:\n' + '\t'.join(
            [kp[0] + ' ' + str(kp[1]) for kp
             in sorted(not_matched, reverse=True, key=lambda kp: kp[1])]))
        return suggestions

    def _keyphrase2uris(self, keyphrase):
        keyphrase = self._normalize_phrase(keyphrase)
        keyphrase = self._sort_phrase(keyphrase)
        return self._index.get(keyphrase, [])

    def _transform_score(self, score):
        score = max(score, 0)
        return 1.0 / (score + 1)

    def _combine_suggestions(self, suggestions):
        combined_suggestions = {}
        for uri, score in suggestions:
            if uri not in combined_suggestions:
                combined_suggestions[uri] = score
            else:
                old_score = combined_suggestions[uri]
                combined_suggestions[uri] = self._combine_scores(
                    score, old_score)
        return list(combined_suggestions.items())

    def _combine_scores(self, score1, score2):
        # The result is never smaller than the greater input
        score1 = score1/2 + 0.5
        score2 = score2/2 + 0.5
        confl = score1 * score2 / (score1 * score2 + (1-score1) * (1-score2))
        return (confl-0.5) * 2

    def _train(self, corpus, params, jobs=0):
        raise NotSupportedException(
            'Training yake backend is not possible.')
