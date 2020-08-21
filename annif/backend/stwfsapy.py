import os
from rdflib import Graph
from rdflib.util import guess_format
from stwfsapy.predictor import StwfsapyPredictor
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import ListSuggestionResult, SubjectSuggestion
from . import backend
from annif.util import boolean


_KEY_GRAPH_PATH = 'graph_path'
_KEY_CONCEPT_TYPE_URI = 'concept_type_uri'
_KEY_SUBTHESAURUS_TYPE_URI = 'sub_thesaurus_type_uri'
_KEY_THESAURUS_RELATION_TYPE_URI = 'thesaurus_relation_type_uri'
_KEY_THESAURUS_RELATION_IS_SPECIALISATION = (
    'thesaurus_relation_is_specialisation')
_KEY_REMOVE_DEPRECATED = 'remove_deprecated'
_KEY_HANDLE_TITLE_CASE = 'handle_title_case'
_KEY_EXTRACT_UPPER_CASE_FROM_BRACES = 'extract_upper_case_from_braces'
_KEY_EXTRACT_ANY_CASE_FROM_BRACES = 'extract_any_case_from_braces'
_KEY_EXPAND_AMPERSAND_WITH_SPACES = 'expand_ampersand_with_spaces'
_KEY_EXPAND_ABBREVIATION_WITH_PUNCTUATION = (
    'expand_abbreviation_with_punctuation')
_KEY_SIMPLE_ENGLISH_PLURAL_RULES = 'simple_english_plural_rules'


class StwfsapyBackend(backend.AnnifBackend):

    name = "stwfsapy"
    needs_subject_index = False

    STWFSAPY_PARAMETERS = {
        _KEY_GRAPH_PATH: str,
        _KEY_CONCEPT_TYPE_URI: str,
        _KEY_SUBTHESAURUS_TYPE_URI: str,
        _KEY_THESAURUS_RELATION_TYPE_URI: str,
        _KEY_THESAURUS_RELATION_IS_SPECIALISATION: boolean,
        _KEY_REMOVE_DEPRECATED: boolean,
        _KEY_HANDLE_TITLE_CASE: boolean,
        _KEY_EXTRACT_UPPER_CASE_FROM_BRACES: boolean,
        _KEY_EXTRACT_ANY_CASE_FROM_BRACES: boolean,
        _KEY_EXPAND_AMPERSAND_WITH_SPACES: boolean,
        _KEY_EXPAND_ABBREVIATION_WITH_PUNCTUATION: boolean,
        _KEY_SIMPLE_ENGLISH_PLURAL_RULES: boolean,
    }

    DEFAULT_PARAMETERS = {
        _KEY_THESAURUS_RELATION_IS_SPECIALISATION: False,
        _KEY_REMOVE_DEPRECATED: True,
        _KEY_HANDLE_TITLE_CASE: True,
        _KEY_EXTRACT_UPPER_CASE_FROM_BRACES: True,
        _KEY_EXTRACT_ANY_CASE_FROM_BRACES: False,
        _KEY_EXPAND_AMPERSAND_WITH_SPACES: True,
        _KEY_EXPAND_ABBREVIATION_WITH_PUNCTUATION: True,
        _KEY_SIMPLE_ENGLISH_PLURAL_RULES: False,
    }

    MODEL_FILE = 'stwfsapy_predictor.zip'

    _model = None

    def initialize(self):
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            self.debug(f'Loading STWFSAPY model from {path}.')
            if os.path.exists(path):
                self._model = StwfsapyPredictor.load(path)
                self.debug('Loaded model.')
            else:
                raise NotInitializedException(
                    f'Model not found at {path}',
                    backend_id=self.backend_id)

    def _train(self, corpus, params):
        if corpus == 'cached':
            raise NotSupportedException(
                'Training stwfsapy project from cached data not supported.')
        if corpus.is_empty():
            raise NotSupportedException(
                'Cannot train stwfsapy project with no documents.')
        self.debug("Transforming training data.")
        X = [doc.text for doc in corpus.documents]
        y = [doc.uris for doc in corpus.documents]
        graph = Graph()
        graph_path = params[_KEY_GRAPH_PATH]
        graph.load(graph_path, format=guess_format(graph_path))
        new_params = {
                key: self.STWFSAPY_PARAMETERS[key](val)
                for key, val
                in params.items()
                if key in self.STWFSAPY_PARAMETERS
            }
        new_params.pop(_KEY_GRAPH_PATH)
        p = StwfsapyPredictor(
            graph=graph,
            langs=frozenset([params['language']]),
            **new_params)
        p.fit(X, y)
        self._model = p
        p.store(os.path.join(self.datadir, self.MODEL_FILE))

    def _suggest(self, text, params):
        self.debug(
            f'Suggesting subjects for text "{text[:20]}..." (len={len(text)})')
        result = self._model.suggest_proba([text])[0]
        suggestions = []
        for uri, score in result:
            subject_id = self.project.subjects.by_uri(uri)
            if subject_id:
                label = self.project.subjects[subject_id][1]
            else:
                label = None
            suggestion = SubjectSuggestion(
                uri,
                label,
                None,
                score)
            suggestions.append(suggestion)
        return ListSuggestionResult(suggestions)
