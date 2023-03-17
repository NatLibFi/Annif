"""Utility functions for Annif"""

import glob
import logging
import os
import os.path
import tempfile

import numpy as np
import scipy.sparse

from annif import logger
from annif.suggestion import VectorSuggestionResult


class DuplicateFilter(logging.Filter):
    """Filter out log messages that have already been displayed."""

    def __init__(self):
        super().__init__()
        self.logged = set()

    def filter(self, record):
        current_log = hash((record.module, record.levelno, record.msg, record.args))
        if current_log not in self.logged:
            self.logged.add(current_log)
            return True
        return False


def atomic_save(obj, dirname, filename, method=None):
    """Save the given object (which must have a .save() method, unless the
    method parameter is given) into the given directory with the given
    filename, using a temporary file and renaming the temporary file to the
    final name."""

    prefix, suffix = os.path.splitext(filename)
    tempfd, tempfilename = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dirname)
    os.close(tempfd)
    logger.debug("saving %s to temporary file %s", str(obj)[:90], tempfilename)
    if method is not None:
        method(obj, tempfilename)
    else:
        obj.save(tempfilename)
    for fn in glob.glob(tempfilename + "*"):
        newname = fn.replace(tempfilename, os.path.join(dirname, filename))
        logger.debug("renaming temporary file %s to %s", fn, newname)
        os.rename(fn, newname)


def cleanup_uri(uri):
    """remove angle brackets from a URI, if any"""
    if uri.startswith("<") and uri.endswith(">"):
        return uri[1:-1]
    return uri


def merge_hits(weighted_hits_batches, size):
    """Merge hit sets from multiple sources. Input is a sequence of
    WeightedSuggestionsBatch objects. The size parameter determines the length of the
    subject vector. Returns a list of SuggestionResult objects."""

    weights = [batch.weight for batch in weighted_hits_batches]
    score_vectors = np.array(
        [
            [whits.as_vector(size) for whits in batch.hit_sets]
            for batch in weighted_hits_batches
        ]
    )
    results = np.average(score_vectors, axis=0, weights=weights)
    return [VectorSuggestionResult(res) for res in results]


def parse_sources(sourcedef):
    """parse a source definition such as 'src1:1.0,src2' into a sequence of
    tuples (src_id, weight)"""

    sources = []
    totalweight = 0.0
    for srcdef in sourcedef.strip().split(","):
        srcval = srcdef.strip().split(":")
        src_id = srcval[0]
        if len(srcval) > 1:
            weight = float(srcval[1])
        else:
            weight = 1.0
        sources.append((src_id, weight))
        totalweight += weight
    return [(srcid, weight / totalweight) for srcid, weight in sources]


def parse_args(param_string):
    """Parse a string of comma separated arguments such as '42,43,key=abc' into
    a list of positional args [42, 43] and a dict of keyword args {key: abc}"""

    if not param_string:
        return [], {}
    posargs = []
    kwargs = {}
    param_strings = param_string.split(",")
    for p_string in param_strings:
        parts = p_string.split("=")
        if len(parts) == 1:
            posargs.append(p_string)
        elif len(parts) == 2:
            kwargs[parts[0]] = parts[1]
    return posargs, kwargs


def boolean(val):
    """Convert the given value to a boolean True/False value, if it isn't already.
    True values are '1', 'yes', 'true', and 'on' (case insensitive), everything
    else is False."""

    return str(val).lower() in ("1", "yes", "true", "on")


def identity(x):
    """Identity function: return the given argument unchanged"""
    return x


def metric_code(metric):
    """Convert a human-readable metric name into an alphanumeric string"""
    return metric.translate(metric.maketrans(" ", "_", "()"))


def filter_suggestion(preds, limit=None, threshold=0.0):
    """filter a 2D sparse suggestion array (csr_array), retaining only the
    top K suggestions with a score above or equal to the threshold for each
    individual prediction; the rest will be left as zeros"""

    filtered = scipy.sparse.dok_array(preds.shape, dtype=np.float32)
    for row in range(preds.shape[0]):
        arow = preds.getrow(row)
        top_k = arow.data.argsort()[::-1]
        if limit is not None:
            top_k = top_k[:limit]
        for idx in top_k:
            val = arow.data[idx]
            if val < threshold:
                break
            filtered[row, arow.indices[idx]] = val
    return filtered.tocsr()
