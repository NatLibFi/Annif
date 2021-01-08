"""Maui-like Lexical Matching backend"""

import collections
from statistics import mean
from rdflib import URIRef
from rdflib.namespace import SKOS
from sklearn.feature_extraction.text import CountVectorizer
from . import backend

Term = collections.namedtuple('Term', 'subject_id label is_pref')
Match = collections.namedtuple(
    'Match', 'subject_id is_pref n_tokens position ambiguity')
Candidate = collections.namedtuple(
    'Candidate',
    'doc_length subject_id freq is_pref n_tokens ambiguity ' +
    'first_occ last_occ spread')


class TokenSet:
    """Represents a set of tokens (expressed as integer token IDs) that can
    be matched with another set of tokens. A TokenSet can optionally
    be associated with a subject from the vocabulary."""

    def __init__(self, tokens, subject_id=None, is_pref=None):
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


class MLLMBackend(backend.AnnifBackend):
    """Maui-like Lexical Matching backend for Annif"""
    name = "mllm"
    needs_subject_index = True

    _vectorizer = None
    _index = None

    def initialize(self):
        # TODO: load vectorizer
        # TODO: load index
        pass

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
                first_occ=matches[0].position / doc_length,
                last_occ=matches[-1].position / doc_length,
                spread=(matches[-1].position - matches[0].position) / doc_length
            )
            for subject_id, matches in subj_matches.items()]

    def _generate_candidates(self, text):
        sentences = self.project.analyzer.tokenize_sentences(text)
        sent_tokens = self._vectorizer.transform(sentences)
        matches = []

        features = self._vectorizer.get_feature_names()

        for sent_idx, token_matrix in enumerate(sent_tokens):
            tset = TokenSet(token_matrix.nonzero()[1])
            for ts, ambiguity in self._index.search(tset):
                matches.append(Match(subject_id=ts.subject_id,
                                     is_pref=ts.is_pref,
                                     n_tokens=len(ts),
                                     position=sent_idx,
                                     ambiguity=ambiguity))

        return self._conflate_matches(matches, len(sentences))

    def _train(self, corpus, params):
        graph = self.project.vocab.as_graph()
        self.info('starting train')
        terms = []
        for subj_id, (uri, pref, _) in enumerate(self.project.vocab.subjects):
            if pref is None:
                continue  # deprecated subject
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
            tokenizer=self.project.analyzer.tokenize_words
        )
        label_corpus = self._vectorizer.fit_transform((t.label for t in terms))
        # TODO: save vectorizer
        self.info(label_corpus.shape)

        self._index = TokenSetIndex()
        for term, label_matrix in zip(terms, label_corpus):
            tokens = label_matrix.nonzero()[1]
            tset = TokenSet(tokens, term.subject_id, term.is_pref)
            self._index.add(tset)
        # TODO: save index

        # TODO: check for "cached" corpus
        for idx, doc in enumerate(corpus.documents):
            self.info(doc.text)
            for candidate in self._generate_candidates(doc.text):
                self.info(candidate)
            if idx > 5:
                break

    def _suggest(self, text, params):
        pass
