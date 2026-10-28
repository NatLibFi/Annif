"""jieba analyzer for Annif which uses rjieba for Chinese word segmentation"""

from __future__ import annotations

import importlib
import re
import unicodedata

import annif.util

from . import analyzer

_KEY_USERDICT = "userdict"
_KEY_HMM = "hmm"

# Chinese text is not split into sentences by NLTK, so split on full-width
# and ASCII sentence-final punctuation and on line breaks instead. An ASCII
# full stop only ends a sentence after a Chinese character or a full-width
# closing bracket or quote, so decimals (3.5) and abbreviations (U.S.) stay intact.
# Kept as a string so that it is compiled (and cached by re) on first use rather
# than at import time, to keep CLI startup fast.
_SENTENCE_END_RE = r"[。！？；!?;\n]+|(?<=[\u3400-\u4dbf\u4e00-\u9fff）」』”’])\.+"


class JiebaAnalyzer(analyzer.Analyzer):
    """Chinese word segmentation using the jieba algorithm and dictionary, as
    implemented by rjieba (Python bindings for the jieba-rs Rust crate)"""

    name = "jieba"
    # Most Chinese words are one or two characters long, so the default
    # minimum token length of 3 would discard nearly all of them
    token_min_length = 1

    @staticmethod
    def is_available() -> bool:
        # return True iff rjieba is installed
        return importlib.util.find_spec("rjieba") is not None

    def __init__(self, param: str | None, **kwargs) -> None:
        self.param = param
        # a custom dictionary may be given as the positional parameter or
        # as userdict=path; it uses the jieba format: word [freq] [tag]
        # (loaded with Jieba.add_word, which requires rjieba 0.2.2 or newer)
        self.userdict = kwargs.pop(_KEY_USERDICT, None) or param
        self.hmm = annif.util.boolean(kwargs.pop(_KEY_HMM, True))
        self._tokenizer = None
        super().__init__(**kwargs)

    def __getstate__(self) -> dict:
        # The rjieba tokenizer is a native object that cannot be pickled (the
        # Omikuji backend pickles the analyzer along with its vectorizer), so
        # leave it out and recreate it on demand after unpickling
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        return state

    @staticmethod
    def _load_userdict(tokenizer, path: str) -> None:
        with open(path, encoding="utf-8") as userdict:
            for line in userdict:
                word, *rest = line.split() or [None]
                if word is None:
                    continue
                freq = int(rest[0]) if rest and rest[0].isdigit() else None
                tag = rest[-1] if rest and not rest[-1].isdigit() else None
                # Input text is NFKC-normalized before segmentation, and some
                # backends (e.g. via scikit-learn vectorizers) lowercase it
                # first, so add each entry in normalized and lowercased form
                word = unicodedata.normalize("NFKC", word)
                for form in dict.fromkeys((word, word.lower())):
                    tokenizer.add_word(form, freq, tag)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import rjieba

            tokenizer = rjieba.Jieba()
            if self.userdict:
                self._load_userdict(tokenizer, self.userdict)
            self._tokenizer = tokenizer
        return self._tokenizer

    def tokenize_sentences(self, text: str) -> list[str]:
        sentences = (s.strip() for s in re.split(_SENTENCE_END_RE, text))
        return [s for s in sentences if s]

    def tokenize_words(self, text: str, filter: bool = True) -> list[str]:
        text = unicodedata.normalize("NFKC", text)
        return [
            word.lower()
            for word in (w.strip() for w in self.tokenizer.cut(text, hmm=self.hmm))
            if word and (not filter or self.is_valid_token(word))
        ]
