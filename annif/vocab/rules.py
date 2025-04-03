"""Support for exclude/include rules for subject vocabularies"""


def kwargs_to_exclude_uris(kwargs: dict[str, str]) -> set[str]:
    exclude_uris = set()
    for key, value in kwargs.items():
        vals = value.split("|")
        if key == "exclude":
            exclude_uris.update(vals)
    return exclude_uris
