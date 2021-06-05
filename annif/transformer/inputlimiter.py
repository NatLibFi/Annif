# TODO Add docstring
from annif.exception import ConfigurationException
from annif.corpus import TruncatingDocumentCorpus
from . import transformer


class InputLimiter(transformer.AbstractTransformer):

    name = 'limit_input'

    def __init__(self, project, input_limit):
        self.project = project
        self.input_limit = int(input_limit)
        self._validate_value(self.input_limit)

    def transform_text(self, text):
        return text[:self.input_limit]

    def transform_corpus(self, corpus):
        return TruncatingDocumentCorpus(corpus, self.input_limit)

    def _validate_value(self, input_limit):
        if input_limit < 0:
            raise ConfigurationException(
                'input_limit in limit_input transformer cannot be negative',
                project_id=self.project.project_id)
