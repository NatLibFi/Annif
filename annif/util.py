"""Utility functions for Annif"""

import glob
import os
import os.path
from shutil import rmtree
import tempfile
import numpy as np
from annif import logger
from annif.suggestion import VectorSuggestionResult


def atomic_save(obj, dirname, filename, method=None):
    """Save the given object (which must have a .save() method, unless the
    method parameter is given) into the given directory with the given
    filename, using a temporary file and renaming the temporary file to the
    final name.
    The .save() mehod or the function provided in the method argument
    will be called with the path to the temporary file."""

    prefix, suffix = os.path.splitext(filename)
    tempfd, tempfilename = tempfile.mkstemp(
        prefix=prefix, suffix=suffix, dir=dirname)
    os.close(tempfd)
    logger.debug('saving %s to temporary file %s', str(obj)[:90], tempfilename)
    if method is not None:
        method(obj, tempfilename)
    else:
        obj.save(tempfilename)
    for fn in glob.glob(tempfilename + '*'):
        newname = fn.replace(tempfilename, os.path.join(dirname, filename))
        logger.debug('renaming temporary file %s to %s', fn, newname)
        os.rename(fn, newname)


def atomic_save_folder(obj, dirname, method=None):
    """Save the given object (which must have a .save() method, unless the
    method parameter is given) into the given directory,
    using a temporary directory and renaming the temporary directory to the
    final name.
    The .save() mehod or the function provided in the method argument
    will be called with the path to the temporary directory."""

    tldir = os.path.dirname(dirname.rstrip('/'))
    os.makedirs(dirname, exist_ok=tldir)
    tempdir = tempfile.TemporaryDirectory(dir=tldir)
    temp_dir_name = tempdir.name
    target_pth = dirname
    logger.debug(
        'saving %s to temporary file %s', str(obj)[:90],
        temp_dir_name)
    if method is not None:
        method(obj, temp_dir_name)
    else:
        obj.save(temp_dir_name)
    for fn in glob.glob(temp_dir_name + '*'):
        newname = fn.replace(temp_dir_name, target_pth)
        logger.debug('renaming temporary file %s to %s', fn, newname)
        if os.path.isdir(newname):
            rmtree(newname)
        os.replace(fn, newname)


def cleanup_uri(uri):
    """remove angle brackets from a URI, if any"""
    if uri.startswith('<') and uri.endswith('>'):
        return uri[1:-1]
    return uri


def merge_hits(weighted_hits, size):
    """Merge hits from multiple sources. Input is a sequence of WeightedSuggestion
    objects. The size parameter determines the length of the subject vector.
    Returns an SuggestionResult object."""

    weights = [whit.weight for whit in weighted_hits]
    scores = [whit.hits.as_vector(size) for whit in weighted_hits]
    result = np.average(scores, axis=0, weights=weights)
    return VectorSuggestionResult(result)


def parse_sources(sourcedef):
    """parse a source definition such as 'src1:1.0,src2' into a sequence of
    tuples (src_id, weight)"""

    sources = []
    totalweight = 0.0
    for srcdef in sourcedef.strip().split(','):
        srcval = srcdef.strip().split(':')
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
    param_strings = param_string.split(',')
    for p_string in param_strings:
        parts = p_string.split('=')
        if len(parts) == 1:
            posargs.append(p_string)
        elif len(parts) == 2:
            kwargs[parts[0]] = parts[1]
    return posargs, kwargs


def apply_param_parse_config(configs, params):
    """Applies a parsing configuration to a parameter dict."""
    return {
        param: configs[param](val)
        for param, val in params.items()
        if param in configs and val is not None}


def boolean(val):
    """Convert the given value to a boolean True/False value, if it isn't already.
    True values are '1', 'yes', 'true', and 'on' (case insensitive), everything
    else is False."""

    return str(val).lower() in ('1', 'yes', 'true', 'on')


def identity(x):
    """Identity function: return the given argument unchanged"""
    return x


def metric_code(metric):
    """Convert a human-readable metric name into an alphanumeric string"""
    return metric.translate(metric.maketrans(' ', '_', '()'))
