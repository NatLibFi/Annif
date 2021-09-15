"""Unit tests for the language-filter transform in Annif"""

import pytest
import annif.transform

pytest.importorskip("annif.transform.langfilter")


def test_lang_filter(project):
    transf = annif.transform.get_transform("filter_lang", project)
    text = """
        Kansalliskirjasto on kaikille avoin kulttuuriperintöorganisaatio, joka
        palvelee valtakunnallisesti kansalaisia, tiedeyhteisöjä ja muita
        yhteiskunnan toimijoita.
        The National Library of Finland is the oldest and largest scholarly
        library in Finland. It is responsible for the collection, description,
        preservation and accessibility of Finland’s published national heritage
        and the unique collections under its care.
        Nationalbiblioteket är Finlands största och äldsta vetenskapliga
        bibliotek, som ansvarar för utökning, beskrivning, förvaring och
        tillhandahållande av vårt nationella publikationsarv och av sina unika
        samlingar.
        Abc defghij klmnopqr stuwxyz abc defghij klmnopqr stuwxyz.
        Turvaamme Suomessa julkaistun tai Suomea koskevan julkaistun
        kulttuuriperinnön saatavuuden sekä välittämme ja tuotamme
        tietosisältöjä tutkimukselle, opiskelulle, kansalaisille ja
        yhteiskunnalle. Kehitämme palveluja yhteistyössä kirjastojen,
        arkistojen, museoiden ja muiden toimijoiden kanssa.
    """
    text = ' '.join(text.split())
    text_filtered = """
        Kansalliskirjasto on kaikille avoin kulttuuriperintöorganisaatio, joka
        palvelee valtakunnallisesti kansalaisia, tiedeyhteisöjä ja muita
        yhteiskunnan toimijoita.
        Abc defghij klmnopqr stuwxyz abc defghij klmnopqr stuwxyz.
        Turvaamme Suomessa julkaistun tai Suomea koskevan julkaistun
        kulttuuriperinnön saatavuuden sekä välittämme ja tuotamme
        tietosisältöjä tutkimukselle, opiskelulle, kansalaisille ja
        yhteiskunnalle. Kehitämme palveluja yhteistyössä kirjastojen,
        arkistojen, museoiden ja muiden toimijoiden kanssa.
    """
    text_filtered = ' '.join(text_filtered.split())
    assert transf.transform_text(text) == text_filtered


def test_lang_filter_text_min_length(project):
    text = "This is just some non-Finnish text of 52 characters."
    transf = annif.transform.get_transform("filter_lang", project)
    assert transf.transform_text(text) == text
    # Set a short text_min_length to apply language filtering:
    transf = annif.transform.get_transform(
        "filter_lang(text_min_length=50)", project)
    assert transf.transform_text(text) == ""


def test_lang_filter_sentence_min_length(project):
    text = "This is a non-Finnish sentence of 42 chars. And this of 20 chars."
    transf = annif.transform.get_transform(
        "filter_lang(text_min_length=50)", project)
    assert transf.transform_text(text) == text
    # Set a short sentence_min_length to apply language filtering:
    transf = annif.transform.get_transform(
        "filter_lang(text_min_length=50,sentence_min_length=30)", project)
    assert transf.transform_text(text) == "And this of 20 chars."
