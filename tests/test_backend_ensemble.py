"""Unit tests for the ensemble backend in Annif"""

import pytest
import annif.backend
from annif.exception import NotSupportedException


def test_ensemble_train(project, document_corpus):
    ensemble_type = annif.backend.get_backend('ensemble')
    ensemble = ensemble_type(
        backend_id='ensemble',
        config_params={'sources': 'dummy'},
        project=project)

    with pytest.raises(NotSupportedException):
        ensemble.train(document_corpus)
