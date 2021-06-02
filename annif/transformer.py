"""Common functionality for input transforming."""

import abc
import re
import annif
from annif.corpus import TruncatingDocumentCorpus, FilteringDocumentCorpus
from annif.exception import ConfigurationException

logger = annif.logger



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


def get_transformer(transformerspec):
    transformers = parse_specs(transformerspec)
    for (trans, _, _) in transformers:
        if trans not in _transformers:
            raise ConfigurationException(f"No such transformer {trans}")
    return Transformer(transformers)


class Transformer():
    """"""  # TODO

    # name = None

    def __init__(self, transformers=None):
        if transformers is None:
            transformers = []
        self.transformers = [_transformers[trans](*posargs, **kwargs)
                             for trans, posargs, kwargs in transformers]

    def transform_text(self, text):
        for trans in self.transformers:
            text = trans.transform_text(text)
        return text

    def transform_corpus(self, corpus):
        for trans in self.transformers:
            corpus = trans.transform_corpus(corpus)
        return corpus


class InputLimiter():

    def __init__(self, input_limit, *posargs):  # TODO Error on unkonwn params
        input_limit = self._validate_input_limit(input_limit)
        self.input_limit = input_limit

    def transform_text(self, text):
        return text[:self.input_limit]

    def transform_corpus(self, corpus):
        return TruncatingDocumentCorpus(corpus, self.input_limit)

    def _validate_input_limit(self, input_limit):
        input_limit = int(input_limit)
        if input_limit >= 0:
            return input_limit
        else:
            raise ConfigurationException(
                'input_limit can not be negative')  # TODO show project id


class LangFilter():

    def __init__(self, *posargs):  # TODO Error on unkonwn params
        pass

    def transform_text(self, text):
        # print('lang filter in action')
        return text

    def transform_corpus(self, corpus):
        # print('lang filter in action')
        return corpus


_transformers = {'input_limit': InputLimiter, 'filter_lang': LangFilter}
