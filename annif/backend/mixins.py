"""Annif backend mixins that can be used to implement features"""


import abc
from annif.suggestion import ListSuggestionResult


class ChunkingBackend(metaclass=abc.ABCMeta):
    """Annif backend mixin that implements chunking of input"""

    @abc.abstractmethod
    def _suggest_chunks(self, chunktexts, project):
        """Suggest subjects for the chunked text; should be implemented by
        the subclass inheriting this mixin"""

        pass  # pragma: no cover

    def _suggest(self, text, project, params):
        self.debug('Suggesting subjects for text "{}..." (len={})'.format(
            text[:20], len(text)))
        sentences = project.analyzer.tokenize_sentences(text)
        self.debug('Found {} sentences'.format(len(sentences)))
        chunksize = int(params['chunksize'])
        chunktexts = []
        for i in range(0, len(sentences), chunksize):
            chunktexts.append(' '.join(sentences[i:i + chunksize]))
        self.debug('Split sentences into {} chunks'.format(len(chunktexts)))
        if len(chunktexts) == 0:  # no input, empty result
            return ListSuggestionResult(
                hits=[], subject_index=project.subjects)
        return self._suggest_chunks(chunktexts, project)
