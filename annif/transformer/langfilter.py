# TODO Add dcostring
import annif
from . import transformer



class LangFilter(transformer.AbstractTransformer):

    name = 'filter_lang'

    def __init__(self, project):  # TODO Error on unknown params
        self.project = project

    def transform_text(self, text):
        # print('lang filter in action')
        # TODO Implement functionality
        return text

    def transform_corpus(self, corpus):
        # print('lang filter in action')
        # TODO Implement functionality
        return corpus
