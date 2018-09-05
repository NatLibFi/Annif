"""Utility functions for Annif"""

import collections
import glob
import os
import tempfile
from annif import logger
from annif.hit import AnalysisResult


def atomic_save(obj, dirname, filename, method=None):
    """Save the given object (which must have a .save() method, unless the
    method parameter is given) into the given directory with the given
    filename, using a temporary file and renaming the temporary file to the
    final name."""

    tempfd, tempfilename = tempfile.mkstemp(prefix=filename, dir=dirname)
    os.close(tempfd)
    logger.debug('saving %s to temporary file %s', str(obj), tempfilename)
    if method is not None:
        method(obj, tempfilename)
    else:
        obj.save(tempfilename)
    for fn in glob.glob(tempfilename + '*'):
        newname = fn.replace(tempfilename, os.path.join(dirname, filename))
        logger.debug('renaming temporary file %s to %s', fn, newname)
        os.rename(fn, newname)


def localname(uri):
    """return the local name extracted from a URI, i.e. the part after the
    last slash or hash character"""

    return uri.split('/')[-1].split('#')[-1]


def cleanup_uri(uri):
    """remove angle brackets from a URI, if any"""
    if uri.startswith('<') and uri.endswith('>'):
        return uri[1:-1]
    return uri


def merge_hits(weighted_hits):
    """Merge hits from multiple sources. Input is a sequence of WeightedHits
    objects"""
    hits_by_uri = collections.defaultdict(list)
    totalweight = 0.0
    for whit in weighted_hits:
        totalweight += whit.weight
        for hit in whit.hits:
            new_hit = hit._replace(score=hit.score * whit.weight)
            hits_by_uri[hit.uri].append(new_hit)
    merged_hits = []
    for hits in hits_by_uri.values():
        total = sum([hit.score for hit in hits]) / totalweight
        merged_hits.append(hits[0]._replace(score=total))
    return AnalysisResult(merged_hits)


def parse_sources(sourcedef):
    """parse a source definition such as 'src1:1.0,src2' into a sequence of
    tuples (src_id, weight)"""

    sources = []
    for srcdef in sourcedef.strip().split(','):
        srcval = srcdef.strip().split(':')
        src_id = srcval[0]
        if len(srcval) > 1:
            weight = float(srcval[1])
        else:
            weight = 1.0
        sources.append((src_id, weight))
    return sources
