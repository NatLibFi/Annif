"""Common functionality for input transforming."""

import abc
# import annif
from annif.exception import ConfigurationException


class AbstractTransformer(metaclass=abc.ABCMeta):
    """"""""  # TODO

    name = None

    def __init__(self, project, *posargs, **kwargs):
        self.project = project

    def transform_text(self, text):
        """"""  # TODO
        return text  # default is to do nothing, subclasses may override

    def transform_corpus(self, corpus):
        """"""  # TODO
        return corpus  # default is to do nothing, subclasses may override


class Transformer():
    """"""  # TODO

    def __init__(self, transformers, args, project):
        self.transformers = []
        self.project = project
        for trans, (posargs, kwargs) in zip(transformers, args):
            try:
                self.transformers.append(
                    trans(self.project, *posargs, **kwargs))
            except (ValueError, TypeError):
                raise ConfigurationException(
                    f"Invalid arguments to {trans.name} transformer: "
                    f"{posargs}, {kwargs})", project_id=project.project_id)

    def transform_text(self, text):
        for trans in self.transformers:
            text = trans.transform_text(text)
        return text

    def transform_corpus(self, corpus):
        for trans in self.transformers:
            corpus = trans.transform_corpus(corpus)
        return corpus
