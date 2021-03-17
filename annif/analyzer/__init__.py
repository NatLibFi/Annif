"""Collection of language-specific analyzers and analyzer registry for Annif"""

import re
from . import simple
from . import snowball
import annif

_analyzers = {}


def register_analyzer(analyzer):
    _analyzers[analyzer.name] = analyzer


def get_analyzer(analyzerspec):
    match = re.match(r'(\w+)(\((.*)\))?', analyzerspec)
    if match is None:
        raise ValueError(
            "Invalid analyzer specification {}".format(analyzerspec))

    analyzer = match.group(1)
    param_string = match.group(3)
    kwargs = {}
    pos_args = []
    if param_string:
        param_strings = param_string.split(',')
        for p_string in param_strings:
            parts = p_string.split('=')
            if len(parts) == 1:
                pos_args.append(p_string)
            elif len(parts) == 2:
                kwargs[parts[0]] = parts[1]
    if not pos_args:
        pos_args = [None]
    try:
        return _analyzers[analyzer](*pos_args, **kwargs)
    except KeyError:
        raise ValueError("No such analyzer {}".format(analyzer))


register_analyzer(simple.SimpleAnalyzer)
register_analyzer(snowball.SnowballAnalyzer)

# Optional analyzers
try:
    from . import voikko
    register_analyzer(voikko.VoikkoAnalyzer)
except ImportError:
    annif.logger.debug("voikko not available, not enabling voikko analyzer")
