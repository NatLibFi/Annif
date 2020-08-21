import os
from annif.backend import get_backend
import annif.corpus
from annif.backend.stwfsapy import StwfsapyBackend
from annif.exception import NotSupportedException

import pytest

_rdf_file_path = os.path.join(
    os.path.dirname(__file__),
    'corpora',
    'archaeology',
    'yso-archaeology.rdf')

_backend_conf = {
    'graph_path': _rdf_file_path,
    'language': 'fi',
    'concept_type_uri': 'http://www.w3.org/2004/02/skos/core#Concept',
    'sub_thesaurus_type_uri':
        'http://www.w3.org/2004/02/skos/core#Collection',
    'thesaurus_relation_type_uri':
        'http://www.w3.org/2004/02/skos/core#member',
    'thesaurus_relation_is_specialisation': True,
}


def test_stwfsapy_default_params(project):
    stwfsapy_type = get_backend(StwfsapyBackend.name)
    stwfsapy = stwfsapy_type(
        backend_id=StwfsapyBackend.name,
        config_params={},
        project=project
    )
    expected_default_params = {
        'thesaurus_relation_is_specialisation': False,
        'remove_deprecated': True,
        'handle_title_case': True,
        'extract_upper_case_from_braces': True,
        'extract_any_case_from_braces': False,
        'expand_ampersand_with_spaces': True,
        'expand_abbreviation_with_punctuation': True,
        'simple_english_plural_rules': False
    }
    actual_params = stwfsapy.params
    assert expected_default_params == actual_params


def test_stwfsapy_train(document_corpus, project, datadir):
    stwfsapy_type = get_backend(StwfsapyBackend.name)
    stwfsapy = stwfsapy_type(
        backend_id=StwfsapyBackend.name,
        config_params=_backend_conf,
        project=project)
    stwfsapy.train(document_corpus)
    assert stwfsapy._model is not None
    model_file = datadir.join(stwfsapy.MODEL_FILE)
    assert model_file.exists()
    assert model_file.size() > 0


def test_empty_corpus(project):
    corpus = annif.corpus.DocumentList([])
    stwfsapy_type = get_backend(StwfsapyBackend.name)
    stwfsapy = stwfsapy_type(
        backend_id=StwfsapyBackend.name,
        config_params=dict(),
        project=project)
    with pytest.raises(NotSupportedException):
        stwfsapy.train(corpus)


def test_cached_corpus(project):
    corpus = 'cached'
    stwfsapy_type = get_backend(StwfsapyBackend.name)
    stwfsapy = stwfsapy_type(
        backend_id=StwfsapyBackend.name,
        config_params=dict(),
        project=project)
    with pytest.raises(NotSupportedException):
        stwfsapy.train(corpus)


def test_stwfsapy_suggest_unknown(project):
    stwfsapy_type = get_backend(StwfsapyBackend.name)
    stwfsapy = stwfsapy_type(
        backend_id=StwfsapyBackend.name,
        config_params=dict(),
        project=project)
    results = stwfsapy.suggest('1234')
    assert len(results) == 0


def test_stwfsapy_suggest(project, datadir):
    stwfsapy_type = get_backend(StwfsapyBackend.name)
    stwfsapy = stwfsapy_type(
        backend_id=StwfsapyBackend.name,
        config_params=dict(),
        project=project)
    # Just some randomly selected words, taken from YSO archaeology group.
    # And "random" words between them
    results = stwfsapy.suggest("""random
    muinais-DNA random random
    labyrintit random random random
    Eurooppalainen yleissopimus arkeologisen perinnön suojelusta random
    Indus-kulttuuri random random random random
    kiinteät muinaisjäännökset random random
    makrofossiilit random
    Mesa Verde random random random random
    muinaismuistoalueet  random random random
    zikkuratit random random
    termoluminesenssi random random random""")
    assert len(results) == 10
    hits = results.as_list(project.subjects)
    assert 'http://www.yso.fi/onto/yso/p14174' in [
        result.uri for result in hits]
    assert 'labyrintit' in [result.label for result in hits]
