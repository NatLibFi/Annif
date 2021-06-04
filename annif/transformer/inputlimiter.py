# TODO Add docstring
from annif.exception import ConfigurationException
from annif.corpus import TruncatingDocumentCorpus
from . import transformer


class InputLimiter(transformer.AbstractTransformer):

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
