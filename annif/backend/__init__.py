"""Registry of backend types for Annif"""

from . import dummy
from . import ensemble
from . import http
from . import tfidf
from . import pav
import annif


_backend_types = {}


def register_backend(backend):
    _backend_types[backend.name] = backend


def get_backend(backend_id):
    try:
        return _backend_types[backend_id]
    except KeyError:
        raise ValueError("No such backend type {}".format(backend_id))


register_backend(dummy.DummyBackend)
register_backend(ensemble.EnsembleBackend)
register_backend(http.HTTPBackend)
register_backend(tfidf.TFIDFBackend)
register_backend(pav.PAVBackend)

# Optional backends
try:
    from . import fasttext
    register_backend(fasttext.FastTextBackend)
except ImportError:
    annif.logger.debug("fastText not available, not enabling fasttext backend")

try:
    from . import vw_multi
    register_backend(vw_multi.VWMultiBackend)
    from . import vw_ensemble
    register_backend(vw_ensemble.VWEnsembleBackend)
except ImportError:
    annif.logger.debug("vowpalwabbit not available, not enabling " +
                       "vw_multi & vw_ensemble backends")
