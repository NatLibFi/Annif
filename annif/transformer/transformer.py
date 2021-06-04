"""Common functionality for input transforming."""

# import abc
# import annif
# from annif.exception import ConfigurationException
#

class Transformer():
    """"""  # TODO

    # name = None

    def __init__(self, transformers=None):
        if transformers is None:
            transformers = []
        self.transformers = transformers

    def transform_text(self, text):
        for trans in self.transformers:
            text = trans.transform_text(text)
        return text

    def transform_corpus(self, corpus):
        for trans in self.transformers:
            corpus = trans.transform_corpus(corpus)
        return corpus
