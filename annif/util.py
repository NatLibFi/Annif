"""Utility functions for Annif"""

import glob
import os
import tempfile
from annif import logger


def atomic_save(obj, dirname, filename):
    """Save the given object (which must have a .save() method) into the
    given directory with the given filename, using a temporary file and
    renaming the temporary file to the final name."""

    tempfd, tempfilename = tempfile.mkstemp(prefix=filename, dir=dirname)
    os.close(tempfd)
    logger.debug('saving %s to temporary file %s', str(obj), tempfilename)
    obj.save(tempfilename)
    for fn in glob.glob(tempfilename + '*'):
        newname = fn.replace(tempfilename, os.path.join(dirname, filename))
        logger.debug('renaming temporary file %s to %s', fn, newname)
        os.rename(fn, newname)
