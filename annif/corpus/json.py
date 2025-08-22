"""Support for document corpora in JSON format"""

import functools
import json
import os.path
from importlib.resources import files

import jsonschema

import annif
from annif.vocab import SubjectIndex

from .types import Document, SubjectSet

logger = annif.logger


@functools.lru_cache(maxsize=1)
def _get_json_schema(schema_name):
    schema_path = files("annif.schemas").joinpath(schema_name)
    with schema_path.open("r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def _subjects_to_subject_set(subjects, subject_index, language):
    subject_ids = []
    for subj in subjects:
        if "uri" in subj:
            subject_ids.append(subject_index.by_uri(subj["uri"]))
        else:
            subject_ids.append(subject_index.by_label(subj["label"], language))
    return SubjectSet(subject_ids)


def json_to_document(
    filename: str,
    json_data: str,
    subject_index: SubjectIndex | None,
    language: str,
    require_subjects: bool,
) -> Document | None:

    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as err:
        logger.warning(f"JSON parsing failed for file {filename}: {err}")
        return None

    try:
        jsonschema.validate(instance=data, schema=_get_json_schema("document.json"))
    except jsonschema.ValidationError as err:
        logger.warning(f"JSON validation failed for file {filename}: {err.message}")
        return None

    if require_subjects:
        subject_set = _subjects_to_subject_set(
            data.get("subjects", []), subject_index, language
        )
        if not subject_set:
            return None
    else:
        subject_set = None

    return Document(
        text=data.get("text", ""),
        metadata=data.get("metadata", {}),
        subject_set=subject_set,
        file_path=filename,
    )


def json_file_to_document(
    filename: str,
    subject_index: SubjectIndex | None,
    language: str,
    require_subjects: bool,
) -> Document | None:
    if os.path.getsize(filename) == 0:
        logger.warning(f"Skipping empty file {filename}")
        return None

    with open(filename, "r", encoding="utf-8") as jsonfile:
        json_data = jsonfile.read()

    return json_to_document(
        filename, json_data, subject_index, language, require_subjects
    )
