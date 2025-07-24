"""Unit tests for the input-transforms in Annif"""

import pytest

import annif.transform
from annif.corpus import Document
from annif.exception import ConfigurationException
from annif.transform import parse_specs
from annif.transform.transform import BaseTransform


def test_parse_specs():
    parsed = parse_specs("foo, bar(42,43,key=abc)")
    assert parsed == [("foo", [], {}), ("bar", ["42", "43"], {"key": "abc"})]


def test_get_transform_nonexistent():
    with pytest.raises(ConfigurationException):
        annif.transform.get_transform("nonexistent", project=None)


def test_get_transform_badspec(project):
    with pytest.raises(ConfigurationException):
        annif.transform.get_transform("pass(invalid_argument)", project)


def test_input_limiter():
    transf = annif.transform.get_transform("limit(3)", project=None)
    doc = Document(text="running")
    assert transf.transform_doc(doc).text == "run"


def test_input_limiter_with_negative_value(project):
    with pytest.raises(ConfigurationException):
        annif.transform.get_transform("limit(-2)", project)


def test_chained_transforms_doc():
    transf = annif.transform.get_transform("limit(5),pass,limit(3),", project=None)
    assert transf.transform_doc(Document(text="abcdefghij")).text == "abc"

    # Check with a more arbitrary transform function
    reverser = annif.transform.transform.IdentityTransform(None)
    reverser.transform_text = lambda x: x[::-1]
    transf.transforms.append(reverser)
    assert transf.transform_doc(Document(text="abcdefghij")).text == "cba"


def test_chained_transforms_corpus(document_corpus):
    transf = annif.transform.get_transform("limit(5),pass,limit(3),", project=None)
    transformed_corpus = transf.transform_corpus(document_corpus)
    for transf_doc, doc in zip(transformed_corpus.documents, document_corpus.documents):
        assert transf_doc.text == doc.text[:3]
        assert transf_doc.subject_set == doc.subject_set

    # Check with a more arbitrary transform function
    reverser = annif.transform.transform.IdentityTransform(None)
    reverser.transform_text = lambda x: x[::-1]
    transf.transforms.append(reverser)
    for transf_doc, doc in zip(transformed_corpus.documents, document_corpus.documents):
        assert transf_doc.text == doc.text[:3][::-1]
        assert transf_doc.subject_set == doc.subject_set


def test_transform_not_implemented():
    class NotImplementedTransform(BaseTransform):
        pass

    with pytest.raises(NotImplementedError):
        NotImplementedTransform(None)
