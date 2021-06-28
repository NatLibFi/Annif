"""Common functionality for input transforming."""

from annif.corpus import TransformingDocumentCorpus
from annif.exception import ConfigurationException


class IdentityTransformer():
    """"""""  # TODO

    name = 'pass'

    def __init__(self, project):
        self.project = project

    def transform_text(self, text):
        """"""  # TODO
        return text

    def transform_corpus(self, corpus):
        """"""  # TODO
        return TransformingDocumentCorpus(corpus, self.transform_text)


class Transformer():
    """"""  # TODO

    def __init__(self, transformer_classes, args, project):
        self.project = project
        self.transformers = self._init_transformers(transformer_classes, args)

    def _init_transformers(self, transformer_classes, args):
        transformers = []
        for trans, (posargs, kwargs) in zip(transformer_classes, args):
            try:
                transformers.append(
                    trans(self.project, *posargs, **kwargs))
            except (ValueError, TypeError):
                raise ConfigurationException(
                    f"Invalid arguments to {trans.name} transformer: "
                    f"{posargs}, {kwargs})",
                    project_id=self.project.project_id)
        return transformers

    def transform_text(self, text):
        for trans in self.transformers:
            text = trans.transform_text(text)
        return text

    def transform_corpus(self, corpus):
        for trans in self.transformers:
            corpus = trans.transform_corpus(corpus)
        return corpus
