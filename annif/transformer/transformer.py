"""Common functionality for input transforming."""

from annif.corpus import TransformingDocumentCorpus
from annif.exception import ConfigurationException


class IdentityTransform():
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


class TransformChain():
    """"""  # TODO

    def __init__(self, transform_classes, args, project):
        self.project = project
        self.transforms = self._init_transforms(transform_classes, args)

    def _init_transforms(self, transform_classes, args):
        transforms = []
        for trans, (posargs, kwargs) in zip(transform_classes, args):
            try:
                transforms.append(
                    trans(self.project, *posargs, **kwargs))
            except (ValueError, TypeError):
                raise ConfigurationException(
                    f"Invalid arguments to {trans.name} transform: "
                    f"{posargs}, {kwargs})",
                    project_id=self.project.project_id)
        return transforms

    def transform_text(self, text):
        for trans in self.transforms:
            text = trans.transform_text(text)
        return text

    def transform_corpus(self, corpus):
        for trans in self.transforms:
            corpus = trans.transform_corpus(corpus)
        return corpus
