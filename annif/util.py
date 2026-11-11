"""Utility functions for Annif"""

from __future__ import annotations

import glob
import importlib.metadata
import logging
import math
import os
import os.path
import re
import tempfile
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from annif.corpus.subject import SubjectIndex
    from annif.suggestion import SubjectSuggestion, SuggestionResults

from annif import logger


class DuplicateFilter(logging.Filter):
    """Filter out log messages that have already been displayed."""

    def __init__(self) -> None:
        super().__init__()
        self.logged = set()

    def filter(self, record: logging.LogRecord) -> bool:
        current_log = hash((record.module, record.levelno, record.msg, record.args))
        if current_log not in self.logged:
            self.logged.add(current_log)
            return True
        return False


def atomic_save(
    obj: Any, dirname: str, filename: str, method: Callable | None = None
) -> None:
    """Save the given object (which must have a .save() method, unless the
    method parameter is given) into the given directory with the given
    filename, using a temporary file and renaming the temporary file to the
    final name."""

    prefix, suffix = os.path.splitext(filename)
    prefix = "tmp-" + prefix
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
        umask = os.umask(0o777)
        os.umask(umask)
        os.chmod(newname, 0o666 & ~umask)


def cleanup_uri(uri: str) -> str:
    """remove angle brackets from a URI, if any"""
    if uri.startswith("<") and uri.endswith(">"):
        return uri[1:-1]
    return uri


@lru_cache
def annif_version() -> str:
    """Return the installed Annif version, looked up once and cached."""
    return importlib.metadata.version("annif")


def user_agent_header() -> dict[str, str]:
    """Return a fresh HTTP headers dict identifying the Annif version,
    e.g. {'User-Agent': 'Annif/1.5.0'}."""
    return {"User-Agent": f"Annif/{annif_version()}"}


# Weight used for a source whose configured weight is zero, negative or not finite
MIN_SOURCE_WEIGHT = 0.01


def parse_sources(sourcedef: str) -> list[tuple[str, float]]:
    """parse a source definition such as 'src1:1.0,src2' into a sequence of
    tuples (src_id, weight). The weights are normalized to sum to 1. A weight
    that is zero, negative or not finite (nan or infinite) is replaced by
    MIN_SOURCE_WEIGHT, with a warning. A ConfigurationException is raised if a
    weight is not a number at all."""
    from annif.exception import ConfigurationException

    sources = []
    totalweight = 0.0
    for srcdef in sourcedef.strip().split(","):
        srcval = srcdef.strip().split(":")
        src_id = srcval[0]
        if len(srcval) > 1:
            try:
                weight = float(srcval[1])
            except ValueError:
                raise ConfigurationException(
                    f"weight '{srcval[1]}' in sources '{sourcedef}' is not a number"
                ) from None
            if not math.isfinite(weight) or weight <= 0:
                logger.warning(
                    "weight %s for source '%s' in sources '%s' is not a positive "
                    "finite number, using %s instead",
                    srcval[1],
                    src_id,
                    sourcedef,
                    MIN_SOURCE_WEIGHT,
                )
                weight = MIN_SOURCE_WEIGHT
        else:
            weight = 1.0
        sources.append((src_id, weight))
        totalweight += weight
    return [(srcid, weight / totalweight) for srcid, weight in sources]


def parse_args(param_string: str) -> tuple[list, dict]:
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


def boolean(val: Any) -> bool:
    """Convert the given value to a boolean True/False value, if it isn't already.
    True values are '1', 'yes', 'true', and 'on' (case insensitive), everything
    else is False."""

    return str(val).lower() in ("1", "yes", "true", "on")


def identity(x: Any) -> Any:
    """Identity function: return the given argument unchanged"""
    return x


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_]\w*)\}", re.ASCII)


def interpolate_env_vars(value: str, project_id: str, option_name: str) -> str:
    """Substitute ${VAR} placeholders in the given string with the value of
    the corresponding environment variable. Other uses of the dollar sign are
    left unchanged. If a referenced environment variable is not set, a
    ConfigurationException is raised."""
    from annif.exception import ConfigurationException

    def _replace(match: re.Match) -> str:
        varname = match.group(1)
        try:
            return os.environ[varname]
        except KeyError:
            raise ConfigurationException(
                f"environment variable '{varname}' referenced in option "
                f"'{option_name}' is not set",
                project_id=project_id,
            ) from None

    return _ENV_VAR_PATTERN.sub(_replace, value)


def metric_code(metric):
    """Convert a human-readable metric name into an alphanumeric string"""
    return metric.translate(metric.maketrans(" ", "_", "()"))


def suggestion_to_dict(
    suggestion: SubjectSuggestion, subject_index: SubjectIndex, language: str
) -> dict[str, str | float | None]:
    subject = subject_index[suggestion.subject_id]
    return {
        "uri": subject.uri,
        "label": subject.labels[language],
        "notation": subject.notation,
        "score": suggestion.score,
    }


def suggestion_results_to_list(
    suggestion_results: SuggestionResults, subjects: SubjectIndex, lang: str
) -> list[dict[str, list]]:
    return [
        {
            "results": [
                suggestion_to_dict(suggestion, subjects, lang)
                for suggestion in suggestions
            ]
        }
        for suggestions in suggestion_results
    ]
