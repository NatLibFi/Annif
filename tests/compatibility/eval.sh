#!/bin/bash

set -x

annif eval tfidf-fi $CORPUS_PATH
annif eval fasttext-fi $CORPUS_PATH
annif eval omikuji-parabel-en $CORPUS_PATH
annif eval mllm-fi $CORPUS_PATH
# annif eval stwfsa-sv $CORPUS_PATH  # Skip evaluating stwfsa until fix #718
annif eval yake-fi $CORPUS_PATH
annif eval svc-en $CORPUS_PATH
annif eval nn-ensemble-fi $CORPUS_PATH
annif eval ensemble-fi $CORPUS_PATH
