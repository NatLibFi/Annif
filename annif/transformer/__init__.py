# TODO Add docstring
import re
from . import transformer
from . import inputlimiter
from . import langfilter
from annif.exception import ConfigurationException


def _extend_posargs(posargs):
    if not posargs:
        posargs = [None]
    return posargs


def _parse_analyzer_args(param_string):
    if not param_string:
        return [None], {}
    kwargs = {}
    posargs = []
    param_strings = param_string.split(',')
    for p_string in param_strings:
        parts = p_string.split('=')
        if len(parts) == 1:
            posargs.append(p_string)
        elif len(parts) == 2:
            kwargs[parts[0]] = parts[1]
    return _extend_posargs(posargs), kwargs


def parse_specs(transformers_spec):
    """parse a configuration definition such as 'A(x),B(y=1),C' into a tuples
    of ((A, [x], {}), (B, [None], {y: 1}))..."""  # TODO

    out = []
    # Split by commas not inside parentheses
    parts = re.split(r',\s*(?![^()]*\))', transformers_spec)
    for part in parts:
        match = re.match(r'(\w+)(\((.*)\))?', part)
        transformer = match.group(1)
        posargs, kwargs = _parse_analyzer_args(match.group(3))
        out.append((transformer, posargs, kwargs))
    return out


def get_transformer(transformers_spec):
    transformers = parse_specs(transformers_spec)
    for (trans, _, _) in transformers:
        if trans not in _transformers:
            raise ConfigurationException(f"No such transformer {trans}")
    return transformer.Transformer(
        [_transformers[trans](*posargs, **kwargs)
         for trans, posargs, kwargs in transformers])


_transformers = {'input_limit': inputlimiter.InputLimiter,
                 'filter_lang': langfilter.LangFilter}
