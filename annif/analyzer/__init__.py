"""Collection of language-specific analyzers and analyzer registry for Annif"""

_analyzers = {}

def register_analyzer(analyzer):
    _analyzers[analyzer.name] = analyzer()

def get_analyzer(analyzer):
    try:
        return _analyzers[analyzer]
    except KeyError:
        raise ValueError("No such analyzer {}".format(analyzer))

from . import english
register_analyzer(english.EnglishAnalyzer)

from . import swedish
register_analyzer(swedish.SwedishAnalyzer)
