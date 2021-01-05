"""Annif backend using Yake keyword extraction"""
# TODO Mention GPLv3 license also here?

import yake
import os.path
import re
from collections import defaultdict
from rdflib.namespace import SKOS, RDF, OWL, URIRef
import rdflib
from nltk.corpus import stopwords
from . import backend
from annif.suggestion import SubjectSuggestion, ListSuggestionResult


class YakeBackend(backend.AnnifBackend):
    """Yake based backend for Annif"""
    name = "yake"
    needs_subject_index = False

    # defaults for uninitialized instances
    _index = None
    _graph = None
    INDEX_FILE = 'yake-index'

    DEFAULT_PARAMETERS = {
        'max_ngram_size': 3,
        'deduplication_threshold': 0.9,
        'deduplication_algo': 'levs',
        'window_size': 1,
        'num_keywords': 100,
        'features': None,
    }

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)
        return params

    @property
    def is_trained(self):
        return True

    def initialize(self):
        self._initialize_index()
        # self.graph
        self._kw_extractor = yake.KeywordExtractor(
            lan=self.project.language,
            n=self.params['max_ngram_size'],
            dedupLim=self.params['deduplication_threshold'],
            dedupFunc=self.params['deduplication_algo'],
            windowsSize=self.params['window_size'],
            top=self.params['num_keywords'],
            features=self.params['features'])

    def _initialize_index(self):
        if self._index is None:
            path = os.path.join(self.datadir, self.INDEX_FILE)
            if os.path.exists(path):
                self.info('Loading index from {}'.format(path))
                self._index = self._load_index(path)
                self.info(f'Loaded index with {len(self._index)} labels')
            else:
                self.info('Creating index')
                self._create_index()
                self._save_index(path)
                self.info(f'Created index with {len(self._index)} labels')

    @property
    def graph(self):
        if self._graph is None:
            self._graph = rdflib.Graph()
            path = os.path.join(self.project.vocab.datadir, 'subjects.ttl')
            self.info('Loading graph from {}'.format(path))
            self._graph.load(path, format=rdflib.util.guess_format(path))
        return self._graph

    def _create_index(self):
        # TODO Should index creation be done on loadvoc command?
        # TODO American to British labels?
        index = defaultdict(list)
        for predicate in [SKOS.prefLabel]:  #, SKOS.altLabel, SKOS.hiddenLabel]:
            for concept in self.graph.subjects(RDF.type, SKOS.Concept):
                if (concept, OWL.deprecated, rdflib.Literal(True)) in self.graph:
                    continue
                for label in self.graph.objects(concept, predicate):
                    if not label.language == self.project.language:
                        continue
                    uri = str(concept)
                    label = str(label)
                    # This really is useful: Disambiguate by dropping ambigious labels
                    # if label[-1] == ')':
                        # continue
                    # label = re.sub(r' \(.*\)', '', label)  # Remove specifier
                    lemmatized_label = self._lemmatize_phrase(label)
                    lemmatized_label = self._sort_phrase(lemmatized_label)
                    index[lemmatized_label].append(uri)
        index.pop('', None)  # Remove possible empty string entry
        self._index = dict(index)

    def _save_index(self, path):
        with open(path, 'w', encoding='utf-8') as indexfile:
            for label, uris in self._index.items():
                line = label + '\t' + ' '.join(uris)
                print(line, file=indexfile)

    def _load_index(self, path):
        index = dict()
        with open(path, 'r', encoding='utf-8') as indexfile:
            for line in indexfile:
                label, uris = line.strip().split('\t')
                uris = uris.split()
                index[label] = uris
        return index

    def _sort_phrase(self, phrase):
        words = phrase.split()
        return ' '.join(sorted(words))

    def _lemmatize_phrase(self, phrase):
        # if self.project.language == 'fi':
            # lan_stopwords = set(stopwords.words('finnish'))
        # elif self.project.language == 'en':
            # stopwords = set(stopwords.words('english'))
        normalized = []
        # phrase = re.sub(r'\W+', '', phrase)
        for word in phrase.split():
            # if word in lan_stopwords:
                # continue
            normalized.append(
                self.project.analyzer.normalize_word(word).lower())
        return ' '.join(normalized)

    def _sort_phrase(self, phrase):
        words = phrase.split()
        return ' '.join(sorted(words))

    def _keyphrases2suggestions(self, keyphrases):
        suggestions = []
        not_matched = []
        for kp, score in keyphrases:
            uris = self._keyphrase2uris(kp)
            for uri in uris:
                # Its faster to get label from Annif subject index than from graph (but is even this needed?)
                label = self.project.subjects.uris_to_labels([uri])[0]
                suggestions.append(
                    (uri, label, self._transform_score(score)))
            if not uris:
                not_matched.append((kp, self._transform_score(score)))
        # Remove duplicate uris, combining the scores
        suggestions = self._combine_suggestions(suggestions)
        self.debug('Keyphrases not matched:\n' + '\t'.join(
            [x[0] + ' ' + str(x[1]) for x
             in sorted(not_matched, reverse=True, key=lambda x: x[1])]))
        return suggestions

    def _keyphrase2uris(self, keyphrase):
        keyphrase = self._lemmatize_phrase(keyphrase)
        keyphrase = self._sort_phrase(keyphrase)
        uris = []
        uris.extend(self._index.get(keyphrase, []))

        # Maybe TODO: Search only in hidden labels if not found in pref or alt labels:
        # if not uris:
            # uris.extend(hidden_label_index.get(mutated_kp, []))

        # Maybe TODO: if not found, search for part of keyword:
        # if not uris and ' ' in keyphrase:
            # words = keyphrase.split()
            # uris.extend(self._index.get(' '.join(words[:-1]), []))
            # uris.extend(self._index.get(' '.join(words[1:]), []))
        return uris

    def _transform_score(self, score):
        return 1.0 / (3*score + 1)

    def _combine_suggestions(self, suggestions):
        combined_suggestions = {}
        for uri, label, score in suggestions:
            if uri not in combined_suggestions:
                combined_suggestions[uri] = (label, score)
            else:
                old_score = combined_suggestions[uri][1]
                conflated_score = self._conflate_scores(score, old_score)
                combined_suggestions[uri] = (label, conflated_score)
        combined_suggestions = [(uri, *label_score) for uri, label_score
                                in combined_suggestions.items()]
        return combined_suggestions

    def _conflate_scores(self, score1, score2):
        # https://stats.stackexchange.com/questions/194878/combining-two-probability-scores/194884
        # return min(1, score1 + score2)
        # return min(1.0, (score1**2 + score2**2)**0.5)
        # score1 = 0.5 * score1 + 0.5
        # score2 = 0.5 * score2 + 0.5
        return score1 * score2 / (score1 * score2 + (1-score1) * (1-score2))

    # def _get_node_degrees(self, suggestions):
    #     connections = []
    #     for uri, label, score in suggestions:
    #         suggestion_neighbours = []
    #         u = URIRef(uri)
    #         suggestion_neighbours.extend(
    #             [o for o in self.graph.objects(u, SKOS.broader)])
    #         suggestion_neighbours.extend(
    #             [o for o in self.graph.objects(u, SKOS.narrower)])
    #         #suggestion_neighbours.extend([o for o in graph.objects(u, SKOS.related)])
    #         connections.append((u, suggestion_neighbours))

    #     node_degrees = []
    #     for uri, label, score in suggestions:
    #         u = URIRef(uri)
    #         cnt = 0
    #         for neighbour, suggestion_neighbours in connections:
    #             if u == neighbour:
    #                 # print('SELF')
    #                 continue
    #             if u in suggestion_neighbours:
    #                 # print('HIT')
    #                 cnt += 1
    #         node_degrees.append(cnt)  # / len(suggestion_neighbours))
    #     return node_degrees

    # def _modify_scores(self, suggestions, node_degrees, scale):
    #     modified_suggestions = []
    #     for suggestion, node_degree in zip(suggestions, node_degrees):
    #         modified_suggestions.append(
    #             (suggestion[0], suggestion[1],
    #              float(suggestion[2]) + scale * node_degree))
    #     return modified_suggestions

    def _suggest(self, text, params):
        self.debug(
            f'Suggesting subjects for text "{text[:20]}..." (len={len(text)})')
        limit = int(params['limit'])

        keywords = self._kw_extractor.extract_keywords(text)
        suggestions = self._keyphrases2suggestions(keywords)

        # node_degrees = self._get_node_degrees(suggestions)
        # suggestions = self._modify_scores(suggestions, node_degrees, scale=0.01)

        subject_suggestions = [SubjectSuggestion(
                uri=uri,
                label=label,
                notation=None,  # TODO Should notation be fetched to here?
                score=score)
                for uri, label, score in suggestions[:limit] if score > 0.0]
        return ListSuggestionResult.create_from_index(subject_suggestions,
                                                      self.project.subjects)
