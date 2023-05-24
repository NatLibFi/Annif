"""Collection of language-specific analyzers and analyzer registry for Annif"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import annif
from annif.util import parse_args

from . import simple, simplemma, snowball

if TYPE_CHECKING:
    from annif.analyzer.analyzer import Analyzer

_analyzers = {}


def register_analyzer(analyzer):
    _analyzers[analyzer.name] = analyzer


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

# Optional analyzers
try:
    from . import voikko

    register_analyzer(voikko.VoikkoAnalyzer)
except ImportError:
    annif.logger.debug("voikko not available, not enabling voikko analyzer")

try:
    from . import spacy

    register_analyzer(spacy.SpacyAnalyzer)
except ImportError:
    annif.logger.debug("spaCy not available, not enabling spacy analyzer")
