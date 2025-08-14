"""Support for document corpora in JSON format"""

import json
import os.path

import annif
from annif.vocab import SubjectIndex

from .types import Document, SubjectSet

logger = annif.logger


def _subjects_to_subject_set(subjects, subject_index, language):
    subject_ids = []
    for subj in subjects:
        if "uri" in subj:
            subject_ids.append(subject_index.by_uri(subj["uri"]))
        else:
            subject_ids.append(subject_index.by_label(subj["label"], language))
    return SubjectSet(subject_ids)


def json_file_to_document(
    filename: str,
    subject_index: SubjectIndex,
    language: str,
    require_subjects: bool,
) -> Document | None:
    if os.path.getsize(filename) == 0:
        logger.warning(f"Skipping empty file {filename}")
        return None

    with open(filename) as jsonfile:
        try:
            data = json.load(jsonfile)
        except json.JSONDecodeError as err:
            logger.warning(f"JSON parsing failed for file {filename}: {err}")
            return None

    subject_set = _subjects_to_subject_set(
        data.get("subjects", []), subject_index, language
    )
    if require_subjects and not subject_set:
        return None

    return Document(
        text=data.get("text", ""),
        metadata=data.get("metadata", {}),
        subject_set=subject_set,
    )
