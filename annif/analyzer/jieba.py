"""jieba analyzer for Annif which uses jieba for Chinese word segmentation"""

from __future__ import annotations

import importlib
import logging
import re
import unicodedata

import annif.util

from . import analyzer

_KEY_USERDICT = "userdict"
_KEY_HMM = "hmm"

# Chinese text is not split into sentences by NLTK, so split on full-width
# and ASCII sentence-final punctuation and on line breaks instead
_SENTENCE_END_RE = re.compile(r"[。！？；!?;\n]+")


class JiebaAnalyzer(analyzer.Analyzer):
    name = "jieba"
    # Most Chinese words are one or two characters long, so the default
    # minimum token length of 3 would discard nearly all of them
    token_min_length = 1

    @staticmethod
    def is_available() -> bool:
        # return True iff jieba is installed
        return importlib.util.find_spec("jieba") is not None

    def __init__(self, param: str | None, **kwargs) -> None:
        self.param = param
        # a custom dictionary may be given as the positional parameter or
        # as userdict=path; one word per line, see the jieba documentation
        self.userdict = kwargs.pop(_KEY_USERDICT, None) or param
        self.hmm = annif.util.boolean(kwargs.pop(_KEY_HMM, True))
        self._tokenizer = None
        super().__init__(**kwargs)

    def __getstate__(self) -> dict:
        # The jieba tokenizer holds a lock and cannot be pickled (the Omikuji
        # backend pickles the analyzer along with its vectorizer), so leave it
        # out and recreate it on demand after unpickling
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        return state

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import jieba

            jieba.setLogLevel(logging.WARNING)
            tokenizer = jieba.Tokenizer()
            tokenizer.initialize()
            if self.userdict:
                tokenizer.load_userdict(self.userdict)
            self._tokenizer = tokenizer
        return self._tokenizer

    def tokenize_sentences(self, text: str) -> list[str]:
        sentences = (s.strip() for s in _SENTENCE_END_RE.split(text))
        return [s for s in sentences if s]

    def tokenize_words(self, text: str, filter: bool = True) -> list[str]:
        text = unicodedata.normalize("NFKC", text)
        return [
            word.lower()
            for word in (w.strip() for w in self.tokenizer.cut(text, HMM=self.hmm))
            if word and (not filter or self.is_valid_token(word))
        ]
