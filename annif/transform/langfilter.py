"""Transformation filtering out parts of a text that are in a language
different from the language of the project."""

import annif
from . import transform

logger = annif.logger


class LangFilter(transform.BaseTransform):

    name = 'filter_lang'

    def __init__(self, project, sentence_min_length=50):
        super().__init__(project)
        self.sentence_min_length = int(sentence_min_length)

    def transform_fn(self, text):
        retained_sentences = []
        sentences = self.project.analyzer.tokenize_sentences(text)
        logger.debug(f'Number of input chars {len(text)} ' +
                     f'in {len(sentences)} sentences')
        for sent in sentences:
            if len(sent) < self.sentence_min_length:
                retained_sentences.append(sent)
                continue
            detected_lang, probability = annif.util.detect_language(sent)
            if detected_lang == self.project.language or detected_lang is None:
                retained_sentences.append(sent)
        text_out = ' '.join(retained_sentences)
        logger.debug(f'Number of retained chars {len(text_out)} ' +
                     f'in {len(retained_sentences)} sentences')
        return text_out
