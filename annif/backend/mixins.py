"""Annif backend mixins that can be used to implement features"""


import abc
from annif.hit import ListAnalysisResult


class ChunkingBackend(metaclass=abc.ABCMeta):
    """Annif backend mixin that implements chunking of input"""

    @abc.abstractmethod
    def _analyze_chunks(self, chunktexts, project):
        """Analyze the chunked text; should be implemented by the subclass
        inheriting this mixin"""

        pass  # pragma: no cover

    def _analyze(self, text, project, params):
        self.initialize()
        self.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        sentences = project.analyzer.tokenize_sentences(text)
        self.debug('Found {} sentences'.format(len(sentences)))
        chunksize = int(params['chunksize'])
        chunktexts = []
        for i in range(0, len(sentences), chunksize):
            chunktexts.append(' '.join(sentences[i:i + chunksize]))
        self.debug('Split sentences into {} chunks'.format(len(chunktexts)))
        if len(chunktexts) == 0:  # nothing to analyze, empty result
            return ListAnalysisResult(hits=[], subject_index=project.subjects)
        return self._analyze_chunks(chunktexts, project)
