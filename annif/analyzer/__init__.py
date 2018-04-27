"""Collection of language-specific analyzers and analyzer registry for Annif"""

import re
from . import simple
from . import snowball

_analyzers = {}


def register_analyzer(analyzer):
    _analyzers[analyzer.name] = analyzer


def get_analyzer(analyzerspec):
    match = re.match(r'(\w+)(\((.*)\))?', analyzerspec)
    if match is None:
        raise ValueError(
            "Invalid analyzer specification {}".format(analyzerspec))

    analyzer = match.group(1)
    try:
        param = match.group(3)
    except IndexError:
        param = None

    try:
        return _analyzers[analyzer](param)
    except KeyError:
        raise ValueError("No such analyzer {}".format(analyzer))


register_analyzer(simple.SimpleAnalyzer)
register_analyzer(snowball.SnowballAnalyzer)
