"""Utility functions for interactions with Hugging Face Hub."""

import binascii
import configparser
import importlib
import io
import os
import pathlib
import shutil
import tempfile
import time
import zipfile
from fnmatch import fnmatch
from typing import Any

import click
from flask import current_app

import annif
from annif import cli_util
from annif.config import AnnifConfigCFG
from annif.exception import OperationFailedException
from annif.project import Access, AnnifProject

logger = annif.logger


def check_is_download_allowed(trust_repo, repo_id):
    """Check if downloading from the specified repository is allowed based on the trust
    option and cache status."""
    if trust_repo:
        logger.warning(
            f'Download allowed from "{repo_id}" because "--trust-repo" flag is used.'
        )
        return
    if _is_repo_in_cache(repo_id):
        logger.debug(
            f'Download allowed from "{repo_id}" because repo is already in cache.'
        )
        return
    raise OperationFailedException(
        f'Cannot download projects from untrusted repo "{repo_id}"'
    )


def _is_repo_in_cache(repo_id):
    from huggingface_hub import CacheNotFound, scan_cache_dir

    try:
        cache = scan_cache_dir()
    except CacheNotFound as err:
        logger.debug(str(err) + "\nNo HFH cache found.")
        return False
    return repo_id in [info.repo_id for info in cache.repos]


def get_matching_projects(pattern: str) -> list[AnnifProject]:
    """
    Get projects that match the given pattern.
    """
    return [
        proj
        for proj in annif.registry.get_projects(min_access=Access.private).values()
        if fnmatch(proj.project_id, pattern)
    ]


def prepare_commits(
    projects: list[AnnifProject], repo_id: str, token: str
) -> tuple[list, list]:
    """Prepare and pre-upload data and config commit operations for projects to a
    Hugging Face Hub repository."""
    from huggingface_hub import preupload_lfs_files

    fobjs, operations = [], []
    data_dirs = {p.datadir for p in projects}
    vocab_dirs = {p.vocab.datadir for p in projects}
    all_dirs = data_dirs.union(vocab_dirs)

    for data_dir in all_dirs:
        fobj, operation = _prepare_datadir_commit(data_dir)
        preupload_lfs_files(repo_id, additions=[operation], token=token)
        fobjs.append(fobj)
        operations.append(operation)

    for project in projects:
        fobj, operation = _prepare_config_commit(project)
        fobjs.append(fobj)
        operations.append(operation)

    return fobjs, operations


def _prepare_datadir_commit(data_dir: str) -> tuple[io.BufferedRandom, Any]:
    from huggingface_hub import CommitOperationAdd

    zip_repo_path = data_dir.split(os.path.sep, 1)[1] + ".zip"
    fobj = _archive_dir(data_dir)
    operation = CommitOperationAdd(path_in_repo=zip_repo_path, path_or_fileobj=fobj)
    return fobj, operation


def _prepare_config_commit(project: AnnifProject) -> tuple[io.BytesIO, Any]:
    from huggingface_hub import CommitOperationAdd

    config_repo_path = project.project_id + ".cfg"
    fobj = _get_project_config(project)
    operation = CommitOperationAdd(path_in_repo=config_repo_path, path_or_fileobj=fobj)
    return fobj, operation


def _is_train_file(fname: str) -> bool:
    train_file_patterns = ("-train", "tmp-")
    for pat in train_file_patterns:
        if pat in fname:
            return True
    return False


def _archive_dir(data_dir: str) -> io.BufferedRandom:
    fp = tempfile.TemporaryFile()
    path = pathlib.Path(data_dir)
    fpaths = [fpath for fpath in path.glob("**/*") if not _is_train_file(fpath.name)]
    with zipfile.ZipFile(fp, mode="w") as zfile:
        zfile.comment = bytes(
            f"Archived by Annif {importlib.metadata.version('annif')}",
            encoding="utf-8",
        )
        for fpath in fpaths:
            logger.debug(f"Adding {fpath}")
            arcname = os.path.join(*fpath.parts[1:])
            zfile.write(fpath, arcname=arcname)
    fp.seek(0)
    return fp


def _get_project_config(project: AnnifProject) -> io.BytesIO:
    fp = tempfile.TemporaryFile(mode="w+t")
    config = configparser.ConfigParser()
    config[project.project_id] = project.config
    config.write(fp)  # This needs tempfile in text mode
    fp.seek(0)
    # But for upload fobj needs to be in binary mode
    return io.BytesIO(fp.read().encode("utf8"))


def get_matching_project_ids_from_hf_hub(
    project_ids_pattern: str, repo_id: str, token, revision: str
) -> list[str]:
    """Get project IDs of the projects in a Hugging Face Model Hub repository that match
    the given pattern."""
    all_repo_file_paths = _list_files_in_hf_hub(repo_id, token, revision)
    return [
        path.rsplit(".cfg")[0]
        for path in all_repo_file_paths
        if fnmatch(path, f"{project_ids_pattern}.cfg")
    ]


def _list_files_in_hf_hub(repo_id: str, token: str, revision: str) -> list[str]:
    from huggingface_hub import list_repo_files
    from huggingface_hub.utils import HfHubHTTPError, HFValidationError

    try:
        return [
            repofile
            for repofile in list_repo_files(
                repo_id=repo_id, token=token, revision=revision
            )
        ]
    except (HfHubHTTPError, HFValidationError) as err:
        raise OperationFailedException(str(err))


def download_from_hf_hub(
    filename: str, repo_id: str, token: str, revision: str
) -> list[str]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError, HFValidationError

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            revision=revision,
        )
    except (HfHubHTTPError, HFValidationError) as err:
        raise OperationFailedException(str(err))


def unzip_archive(src_path: str, force: bool) -> None:
    """Unzip a zip archive of projects and vocabularies to a directory, by
    default data/ under current directory."""
    datadir = current_app.config["DATADIR"]
    with zipfile.ZipFile(src_path, "r") as zfile:
        archive_comment = str(zfile.comment, encoding="utf-8")
        logger.debug(
            f'Extracting archive {src_path}; archive comment: "{archive_comment}"'
        )
        for member in zfile.infolist():
            _unzip_member(zfile, member, datadir, force)


def _unzip_member(
    zfile: zipfile.ZipFile, member: zipfile.ZipInfo, datadir: str, force: bool
) -> None:
    dest_path = os.path.join(datadir, member.filename)
    if os.path.exists(dest_path) and not force:
        _handle_existing_file(member, dest_path)
        return
    logger.debug(f"Unzipping to {dest_path}")
    zfile.extract(member, path=datadir)
    _restore_timestamps(member, dest_path)


def _handle_existing_file(member: zipfile.ZipInfo, dest_path: str) -> None:
    if _are_identical_member_and_file(member, dest_path):
        logger.debug(f"Skipping unzip to {dest_path}; already in place")
    else:
        click.echo(f"Not overwriting {dest_path} (use --force to override)")


def _are_identical_member_and_file(member: zipfile.ZipInfo, dest_path: str) -> bool:
    path_crc = _compute_crc32(dest_path)
    return path_crc == member.CRC


def _restore_timestamps(member: zipfile.ZipInfo, dest_path: str) -> None:
    date_time = time.mktime(member.date_time + (0, 0, -1))
    os.utime(dest_path, (date_time, date_time))


def copy_project_config(src_path: str, force: bool) -> None:
    """Copy a given project configuration file to projects.d/ directory."""
    project_configs_dest_dir = "projects.d"
    os.makedirs(project_configs_dest_dir, exist_ok=True)

    dest_path = os.path.join(project_configs_dest_dir, os.path.basename(src_path))
    if os.path.exists(dest_path) and not force:
        if _are_identical_files(src_path, dest_path):
            logger.debug(f"Skipping copy to {dest_path}; already in place")
        else:
            click.echo(f"Not overwriting {dest_path} (use --force to override)")
    else:
        logger.debug(f"Copying to {dest_path}")
        shutil.copy(src_path, dest_path)


def _are_identical_files(src_path: str, dest_path: str) -> bool:
    src_crc32 = _compute_crc32(src_path)
    dest_crc32 = _compute_crc32(dest_path)
    return src_crc32 == dest_crc32


def _compute_crc32(path: str) -> int:
    if os.path.isdir(path):
        return 0

    size = 1024 * 1024 * 10  # 10 MiB chunks
    with open(path, "rb") as fp:
        crcval = 0
        while chunk := fp.read(size):
            crcval = binascii.crc32(chunk, crcval)
    return crcval


def get_vocab_id_from_config(config_path: str) -> str:
    """Get the vocabulary ID from a configuration file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    section = config.sections()[0]
    return config[section]["vocab"]


def upsert_modelcard(repo_id, projects, token, revision):
    """This function creates or updates a Model Card in a Hugging Face Hub repository
    with some metadata in it."""
    from huggingface_hub import ModelCard
    from huggingface_hub.utils import EntryNotFoundError

    try:
        card = ModelCard.load(repo_id)
        commit_message = "Update README.md with Annif"
    except EntryNotFoundError:
        card = _create_modelcard(repo_id)
        commit_message = "Create README.md with Annif"

    langs_existing = set(card.data.language) if card.data.language else set()
    langs_to_add = {proj.vocab_lang for proj in projects}
    card.data.language = list(langs_existing.union(langs_to_add))

    configs = _get_existing_configs(repo_id, token, revision)
    card.text = _update_projects_section(card.text, configs)

    card.push_to_hub(
        repo_id=repo_id, token=token, revision=revision, commit_message=commit_message
    )


def _get_existing_configs(repo_id, token, revision):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    cfg_locations = fs.glob(f"{repo_id}/*.cfg", revision=revision)

    projstr = ""
    for cfg_file in cfg_locations:
        projstr += fs.read_text(cfg_file, token=token, revision=revision)
    return AnnifConfigCFG(projstr=projstr)


def _create_modelcard(repo_id):
    from huggingface_hub import ModelCard

    content = f"""
---

---

# {repo_id.split("/")[1]}

## Usage

Use the `annif download` command to download selected projects with Annif;
for example, to download all projects in this repository run

    annif download "*" {repo_id}

"""
    card = ModelCard(content)
    card.data.pipeline_tag = "text-classification"
    card.data.tags = ["annif"]
    return card


AUTOUPDATING_START = "<!--- start-of-autoupdating-part --->"
AUTOUPDATING_END = "<!--- end-of-autoupdating-part --->"


def _update_projects_section(text, configs):
    section_start_ind = text.find(AUTOUPDATING_START)
    section_end_ind = text.rfind(AUTOUPDATING_END) + len(AUTOUPDATING_END)

    projects_section = _create_projects_section(configs)
    if section_start_ind == -1:  # no existing projects section, append it now
        return text + projects_section
    else:
        return text[:section_start_ind] + projects_section + text[section_end_ind:]


def _create_projects_section(configs):
    column_headings = (
        "Project ID",
        "Project Name",
        "Vocabulary ID",
        "Language",
    )
    table = [
        (
            proj_id,
            configs[proj_id]["name"],
            configs[proj_id]["vocab"],
            configs[proj_id]["language"],
        )
        for proj_id in configs.project_ids
    ]
    template = cli_util.make_list_template(column_headings, *table) + "\n"

    header = template.format(*column_headings)

    content = f"{AUTOUPDATING_START}\n## Projects\n"
    content += "```\n" + header + "-" * len(header.strip()) + "\n"
    for row in table:
        content += template.format(*row)
    return content + "```\n" + AUTOUPDATING_END
