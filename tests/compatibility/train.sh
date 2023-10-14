#!/bin/bash

set -x

annif load-vocab yso /Annif/tests/corpora/archaeology/yso-archaeology.ttl
annif train tfidf-fi $CORPUS_PATH
annif train fasttext-fi $CORPUS_PATH
annif train omikuji-parabel-en $CORPUS_PATH
annif train mllm-fi $CORPUS_PATH
annif train stwfsa-sv $CORPUS_PATH
annif train svc-en $CORPUS_PATH
annif train nn-ensemble-fi $CORPUS_PATH
