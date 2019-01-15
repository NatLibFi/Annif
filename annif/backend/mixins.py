"""Annif backend mixins that can be used to implement features"""


import abc


class ChunkingBackend(metaclass=abc.ABCMeta):
    """Annif backend mixin that implements chunking of input"""

    @abc.abstractmethod
    def _analyze_chunks(self, chunktexts, project):
        """Analyze the chunked text; should be implemented by the subclass
        inheriting this mixin"""

        pass

    def _analyze(self, text, project, params):
        self.initialize()
        self.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        sentences = project.analyzer.tokenize_sentences(text)
        self.debug('Found {} sentences'.format(len(sentences)))
        chunksize = int(params['chunksize'])
        chunktexts = []
        for i in range(0, len(sentences), chunksize):
            chunktext = ' '.join(sentences[i:i + chunksize])
            normalized = self._normalize_text(project, chunktext)
            if normalized != '':
                chunktexts.append(normalized)
        self.debug('Split sentences into {} chunks'.format(len(chunktexts)))
        return self._analyze_chunks(chunktexts, project)
