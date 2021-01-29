"""Unit tests for the Yake backend in Annif"""

import annif
import pytest
import os
from rdflib import Graph

pytest.importorskip("annif.backend.yake")


@pytest.fixture(scope='module')
def graph_project(project):
    _rdf_file_path = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'yso-archaeology.rdf')
    g = Graph()
    g.load(_rdf_file_path)
    project.vocab.as_graph.return_value = g
    return project


def test_yake_suggest(project, graph_project):
    yake_type = annif.backend.get_backend('yake')
    yake = yake_type(
        backend_id='yake',
        config_params={'limit': 8, 'language': 'fi'},
        project=graph_project)

    results = yake.suggest("""Arkeologia on tieteenala, jota sanotaan joskus
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""")

    assert len(results) > 0
    assert len(results) <= 8
    hits = results.as_list(project.subjects)
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in hits]
    assert 'arkeologia' in [result.label for result in hits]
