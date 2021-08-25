"""Transformation filtering out parts of a text that are in a language
different from the language of the project."""

import annif
from . import transform

logger = annif.logger


class LangFilter(transform.BaseTransform):

    name = 'filter_lang'

    def __init__(self, project, threshold=0.99):
        super().__init__(project)
        self.threshold = float(threshold)

    def transform_fn(self, text):
        retained_sentences = []
        sentences = self.project.analyzer.tokenize_sentences(text)
        logger.debug(f'Number of input chars {len(text)} ' +
                     f'in {len(sentences)} sentences')
        for sent in sentences:
            # TODO: Concat very short sentence to next one for better accuracy?
            detected_lang, probability = annif.util.detect_language(sent)
            # TODO: Retain also those sentences which detection fails on?
            if detected_lang == self.project.language \
                    and probability >= self.threshold:
                retained_sentences.append(sent)
        text_out = ' '.join(retained_sentences)
        logger.debug(f'Number of retained chars {len(text_out)} ' +
                     f'in {len(retained_sentences)} sentences')
        return text_out
