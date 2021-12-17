"""Registry of backend types for Annif"""


# define functions for lazily importing each backend (alphabetical order)
def _dummy():
    from . import dummy
    return dummy.DummyBackend


def _ensemble():
    from . import ensemble
    return ensemble.EnsembleBackend


def _fasttext():
    try:
        from . import fasttext
        return fasttext.FastTextBackend
    except ImportError:
        raise ValueError("fastText not available, cannot use fasttext backend")


def _http():
    from . import http
    return http.HTTPBackend


def _mllm():
    from . import mllm
    return mllm.MLLMBackend


def _nn_ensemble():
    try:
        from . import nn_ensemble
        return nn_ensemble.NNEnsembleBackend
    except ImportError:
        raise ValueError("Keras and TensorFlow not available, cannot use " +
                         "nn_ensemble backend")


def _omikuji():
    try:
        from . import omikuji
        return omikuji.OmikujiBackend
    except ImportError:
        raise ValueError("Omikuji not available, cannot use omikuji backend")


def _pav():
    from . import pav
    return pav.PAVBackend


def _stwfsa():
    from . import stwfsa
    return stwfsa.StwfsaBackend


def _svc():
    from . import svc
    return svc.SVCBackend


def _tfidf():
    from . import tfidf
    return tfidf.TFIDFBackend


def _yake():
    try:
        from . import yake
        return yake.YakeBackend
    except ImportError:
        raise ValueError("YAKE not available, cannot use yake backend")


# registry of the above functions
_backend_fns = {
    'dummy': _dummy,
    'ensemble': _ensemble,
    'fasttext': _fasttext,
    'http': _http,
    'mllm': _mllm,
    'nn_ensemble': _nn_ensemble,
    'omikuji': _omikuji,
    'pav': _pav,
    'stwfsa': _stwfsa,
    'svc': _svc,
    'tfidf': _tfidf,
    'yake': _yake
}


def get_backend(backend_id):
    if backend_id in _backend_fns:
        return _backend_fns[backend_id]()
    else:
        raise ValueError("No such backend type {}".format(backend_id))
