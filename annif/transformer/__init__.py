# TODO Add docstring
import re
from . import transformer
from . import inputlimiter
from .transformer import IdentityTransform
from annif.util import parse_args
from annif.exception import ConfigurationException

__all__ = ["IdentityTransform"]


def parse_specs(transform_specs):
    """parse a configuration definition such as 'A(x),B(y=1),C' into a tuples
    of ((A, [x], {}), (B, [None], {y: 1}))..."""  # TODO
    parsed = []
    # Split by commas not inside parentheses
    parts = re.split(r',\s*(?![^()]*\))', transform_specs)
    for part in parts:
        match = re.match(r'(\w+)(\((.*)\))?', part)
        if match is None:
            continue
        transform = match.group(1)
        posargs, kwargs = parse_args(match.group(3))
        parsed.append((transform, posargs, kwargs))
    return parsed


def get_transform(transform_specs, project):
    transform_defs = parse_specs(transform_specs)
    transform_classes = []
    args = []
    for trans, posargs, kwargs in transform_defs:
        if trans not in _transforms:
            raise ConfigurationException(f"No such transform {trans}")
        transform_classes.append(_transforms[trans])
        args.append((posargs, kwargs))
    return transformer.TransformChain(transform_classes, args, project)


_transforms = {
    transformer.IdentityTransform.name: transformer.IdentityTransform,
    inputlimiter.InputLimiter.name: inputlimiter.InputLimiter}
