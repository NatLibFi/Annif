"""Voikko analyzer for Annif, based on libvoikko library."""
from __future__ import annotations

import functools

import voikko.libvoikko

from . import analyzer


class VoikkoAnalyzer(analyzer.Analyzer):
    name = "voikko"

    def __init__(self, param: str, **kwargs) -> None:
        self.param = param
        self.voikko = None
        super().__init__(**kwargs)

    def __getstate__(self) -> dict[str, str | None]:
        """Return the state of the object for pickling purposes. The Voikko
        instance is set to None because as a ctypes object it cannot be
        pickled."""

        return {"param": self.param, "voikko": None}

    @functools.lru_cache(maxsize=500000)
    def _normalize_word(self, word: str) -> str:
        if self.voikko is None:
            self.voikko = voikko.libvoikko.Voikko(self.param)
        result = self.voikko.analyze(word)
        if len(result) > 0 and "BASEFORM" in result[0]:
            return result[0]["BASEFORM"]
        return word
