# TODO Add docstring
import re
from . import transformer
from . import inputlimiter
from . import langfilter
from annif.exception import ConfigurationException


def _parse_transformer_args(param_string):
    if not param_string:
        return [], {}
    kwargs = {}
    posargs = []
    param_strings = param_string.split(',')
    for p_string in param_strings:
        parts = p_string.split('=')
        if len(parts) == 1:
            posargs.append(p_string)
        elif len(parts) == 2:
            kwargs[parts[0]] = parts[1]
    return posargs, kwargs


def parse_specs(transformers_spec):
    """parse a configuration definition such as 'A(x),B(y=1),C' into a tuples
    of ((A, [x], {}), (B, [None], {y: 1}))..."""  # TODO
    parsed = []
    # Split by commas not inside parentheses
    parts = re.split(r',\s*(?![^()]*\))', transformers_spec)
    for part in parts:
        match = re.match(r'(\w+)(\((.*)\))?', part)
        transformer = match.group(1)
        posargs, kwargs = _parse_transformer_args(match.group(3))
        parsed.append((transformer, posargs, kwargs))
    return parsed


def get_transformer(transformer_specs, project):
    transformer_defs = parse_specs(transformer_specs)
    transformer_classes = []
    args = []
    for trans, posargs, kwargs in transformer_defs:
        if trans not in _transformers:
            raise ConfigurationException(f"No such transformer {trans}")
        transformer_classes.append(_transformers[trans])
        args.append((posargs, kwargs))
    return transformer.Transformer(transformer_classes, args, project)


_transformers = {'limit_input': inputlimiter.InputLimiter,
                 'filter_lang': langfilter.LangFilter}
