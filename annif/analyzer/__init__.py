"""Collection of language-specific analyzers and analyzer registry for Annif"""

analyzers = {}

def register_analyzer(analyzer):
    analyzers[analyzer.name] = analyzer

def get_analyzer(analyzer):
    try:
        return analyzers[analyzer]
    except KeyError:
        raise ValueError("No such analyzer {}".format(analyzer))

