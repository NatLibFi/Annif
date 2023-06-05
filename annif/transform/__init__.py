"""Functionality for obtaining text transformation from string specification"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import annif
from annif.exception import ConfigurationException
from annif.util import parse_args

from . import inputlimiter, transform

if TYPE_CHECKING:
    from annif.project import AnnifProject
    from annif.transform.transform import TransformChain


def parse_specs(
    transform_specs: str,
) -> list[tuple[str, list, dict]]:
    """Parse a transformation specification into a list of tuples, e.g.
    'transf_1(x),transf_2(y=42),transf_3' is parsed to
    [(transf_1, [x], {}), (transf_2, [], {y: 42}), (transf_3, [], {})]."""

    parsed = []
    # Split by commas not inside parentheses
    parts = re.split(r",\s*(?![^()]*\))", transform_specs)
    for part in parts:
        match = re.match(r"(\w+)(\((.*)\))?", part)
        if match is None:
            continue
        transform = match.group(1)
        posargs, kwargs = parse_args(match.group(3))
        parsed.append((transform, posargs, kwargs))
    return parsed


def get_transform(transform_specs: str, project: AnnifProject | None) -> TransformChain:
    transform_defs = parse_specs(transform_specs)
    transform_classes = []
    args = []
    for trans, posargs, kwargs in transform_defs:
        if trans not in _transforms:
            raise ConfigurationException(f"No such transform {trans}")
        transform_classes.append(_transforms[trans])
        args.append((posargs, kwargs))
    return transform.TransformChain(transform_classes, args, project)


_transforms = {
    transform.IdentityTransform.name: transform.IdentityTransform,
    inputlimiter.InputLimiter.name: inputlimiter.InputLimiter,
}

# Optional transforms
try:
    from . import langfilter

    _transforms.update({langfilter.LangFilter.name: langfilter.LangFilter})
except ImportError:
    annif.logger.debug("pycld3 not available, not enabling filter_language transform")
