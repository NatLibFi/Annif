"""Collection of language-specific analyzers and analyzer registry for Annif"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import annif
from annif.util import parse_args

from . import estnltk, simple, simplemma, snowball, spacy, voikko

if TYPE_CHECKING:
    from annif.analyzer.analyzer import Analyzer

_analyzers = {}


def register_analyzer(analyzer):
    if analyzer.is_available():
        _analyzers[analyzer.name] = analyzer
    else:
        annif.logger.debug(f"{analyzer.name} analyzer not available, not enabling it")


def get_analyzer(analyzerspec: str) -> Analyzer:
    match = re.match(r"(\w+)(\((.*)\))?", analyzerspec)
    if match is None:
        raise ValueError("Invalid analyzer specification {}".format(analyzerspec))

    analyzer = match.group(1)
    posargs, kwargs = parse_args(match.group(3))
    posargs = posargs if posargs else [None]
    try:
        return _analyzers[analyzer](*posargs, **kwargs)
    except KeyError:
        raise ValueError("No such analyzer {}".format(analyzer))


register_analyzer(simple.SimpleAnalyzer)
register_analyzer(snowball.SnowballAnalyzer)
register_analyzer(simplemma.SimplemmaAnalyzer)
register_analyzer(voikko.VoikkoAnalyzer)
register_analyzer(spacy.SpacyAnalyzer)
register_analyzer(estnltk.EstNLTKAnalyzer)
