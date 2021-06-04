# TODO Add docstring
from annif.exception import ConfigurationException
from annif.corpus import TruncatingDocumentCorpus
from . import transformer


class InputLimiter(transformer.AbstractTransformer):

    def __init__(self, input_limit):
        self.input_limit = int(input_limit)
        self._validate_value(self.input_limit)

    def transform_text(self, text):
        return text[:self.input_limit]

    def transform_corpus(self, corpus):
        return TruncatingDocumentCorpus(corpus, self.input_limit)

    def _validate_value(self, input_limit):
        if input_limit < 0:
            raise ConfigurationException('input_limit cannot be negative')  # TODO show project id
