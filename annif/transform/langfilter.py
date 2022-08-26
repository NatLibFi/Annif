"""Transformation filtering out parts of a text that are in a language
different from the language of the project."""

import annif
import lingua
from . import transform

logger = annif.logger


class LangFilter(transform.BaseTransform):

    name = 'filter_lang'

    def __init__(self, project, text_min_length=500, sentence_min_length=50):
        super().__init__(project)
        self.text_min_length = int(text_min_length)
        self.sentence_min_length = int(sentence_min_length)
        self.detector = (
            lingua.LanguageDetectorBuilder
            .from_all_languages()
            .with_low_accuracy_mode()
            .build()
        )

    def _detect_language(self, text):
        """Tries to detect the language of a text input. Outputs a BCP-47-style
        language code (e.g. 'en')."""

        lan_info = self.detector.detect_language_of(text)
        if lan_info is not None:
            return lan_info.iso_code_639_1.name.lower()
        else:
            return None

    def transform_fn(self, text):
        if len(text) < self.text_min_length:
            return text

        retained_sentences = []
        for sent in self.project.analyzer.tokenize_sentences(text):
            if len(sent) < self.sentence_min_length:
                retained_sentences.append(sent)
                continue
            detected_lang = self._detect_language(sent)
            if detected_lang == self.project.language or detected_lang is None:
                retained_sentences.append(sent)
        return ' '.join(retained_sentences)
