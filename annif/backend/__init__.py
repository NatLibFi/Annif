"""Registry of backend types for Annif"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    from annif.backend.dummy import DummyBackend
    from annif.backend.ensemble import EnsembleBackend
    from annif.backend.fasttext import FastTextBackend
    from annif.backend.http import HTTPBackend
    from annif.backend.mllm import MLLMBackend
    from annif.backend.nn_ensemble import NNEnsembleBackend
    from annif.backend.omikuji import OmikujiBackend
    from annif.backend.pav import PAVBackend
    from annif.backend.stwfsa import StwfsaBackend
    from annif.backend.svc import SVCBackend
    from annif.backend.tfidf import TFIDFBackend
    from annif.backend.yake import YakeBackend


# define functions for lazily importing each backend (alphabetical order)
def _dummy() -> Type[DummyBackend]:
    from . import dummy

    return dummy.DummyBackend


def _ensemble() -> Type[EnsembleBackend]:
    from . import ensemble

    return ensemble.EnsembleBackend


def _fasttext() -> Type[FastTextBackend]:
    try:
        from . import fasttext

        return fasttext.FastTextBackend
    except ImportError:
        raise ValueError("fastText not available, cannot use fasttext backend")


def _http() -> Type[HTTPBackend]:
    from . import http

    return http.HTTPBackend


def _mllm() -> Type[MLLMBackend]:
    from . import mllm

    return mllm.MLLMBackend


def _nn_ensemble() -> Type[NNEnsembleBackend]:
    try:
        from . import nn_ensemble

        return nn_ensemble.NNEnsembleBackend
    except ImportError:
        raise ValueError(
            "Keras and TensorFlow not available, cannot use " + "nn_ensemble backend"
        )


def _omikuji() -> Type[OmikujiBackend]:
    try:
        from . import omikuji

        return omikuji.OmikujiBackend
    except ImportError:
        raise ValueError("Omikuji not available, cannot use omikuji backend")


def _pav() -> Type[PAVBackend]:
    from . import pav

    return pav.PAVBackend


def _stwfsa() -> Type[StwfsaBackend]:
    try:
        from . import stwfsa

        return stwfsa.StwfsaBackend
    except ImportError:
        raise ValueError("STWFSA not available, cannot use stwfsa backend")


def _svc() -> Type[SVCBackend]:
    from . import svc

    return svc.SVCBackend


def _tfidf() -> Type[TFIDFBackend]:
    from . import tfidf

    return tfidf.TFIDFBackend


def _yake() -> Type[YakeBackend]:
    try:
        from . import yake

        return yake.YakeBackend
    except ImportError:
        raise ValueError("YAKE not available, cannot use yake backend")


# registry of the above functions
_backend_fns = {
    "dummy": _dummy,
    "ensemble": _ensemble,
    "fasttext": _fasttext,
    "http": _http,
    "mllm": _mllm,
    "nn_ensemble": _nn_ensemble,
    "omikuji": _omikuji,
    "pav": _pav,
    "stwfsa": _stwfsa,
    "svc": _svc,
    "tfidf": _tfidf,
    "yake": _yake,
}


def get_backend(backend_id: str) -> Any:
    if backend_id in _backend_fns:
        return _backend_fns[backend_id]()
    else:
        raise ValueError("No such backend type {}".format(backend_id))
