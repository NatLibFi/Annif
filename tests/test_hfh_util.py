"""Unit test module for Hugging Face Hub utilities."""

import io
import os.path
import zipfile
from datetime import datetime, timezone
from unittest import mock

import annif.hfh_util


def test_archive_dir(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    os.makedirs(dirpath, exist_ok=True)
    open(os.path.join(str(dirpath), "foo.txt"), "a").close()
    open(os.path.join(str(dirpath), "-train.txt"), "a").close()

    fobj = annif.hfh_util._archive_dir(dirpath)
    assert isinstance(fobj, io.BufferedRandom)

    with zipfile.ZipFile(fobj, mode="r") as zfile:
        archived_files = zfile.namelist()
    assert len(archived_files) == 1
    assert os.path.split(archived_files[0])[1] == "foo.txt"


def test_get_project_config(app_project):
    result = annif.hfh_util._get_project_config(app_project)
    assert isinstance(result, io.BytesIO)
    string_result = result.read().decode("UTF-8")
    assert "[dummy-en]" in string_result


def test_unzip_archive_initial(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    fpath = os.path.join(str(dirpath), "file.txt")
    annif.hfh_util.unzip_archive(
        os.path.join("tests", "huggingface-cache", "projects", "dummy-fi.zip"),
        force=False,
    )
    assert os.path.exists(fpath)
    assert os.path.getsize(fpath) == 0  # Zero content from zip
    ts = os.path.getmtime(fpath)
    assert datetime.fromtimestamp(ts).astimezone(tz=timezone.utc) == datetime(
        1980, 1, 1, 0, 0
    ).astimezone(tz=timezone.utc)


def test_unzip_archive_no_overwrite(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    fpath = os.path.join(str(dirpath), "file.txt")
    os.makedirs(dirpath, exist_ok=True)
    with open(fpath, "wt") as pf:
        print("Existing content", file=pf)

    annif.hfh_util.unzip_archive(
        os.path.join("tests", "huggingface-cache", "projects", "dummy-fi.zip"),
        force=False,
    )
    assert os.path.exists(fpath)
    assert os.path.getsize(fpath) == 17  # Existing content
    assert datetime.now().timestamp() - os.path.getmtime(fpath) < 1


def test_unzip_archive_overwrite(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    fpath = os.path.join(str(dirpath), "file.txt")
    os.makedirs(dirpath, exist_ok=True)
    with open(fpath, "wt") as pf:
        print("Existing content", file=pf)

    annif.hfh_util.unzip_archive(
        os.path.join("tests", "huggingface-cache", "projects", "dummy-fi.zip"),
        force=True,
    )
    assert os.path.exists(fpath)
    assert os.path.getsize(fpath) == 0  # Zero content from zip
    ts = os.path.getmtime(fpath)
    assert datetime.fromtimestamp(ts).astimezone(tz=timezone.utc) == datetime(
        1980, 1, 1, 0, 0
    ).astimezone(tz=timezone.utc)


@mock.patch("os.path.exists", return_value=True)
@mock.patch("annif.hfh_util._compute_crc32", return_value=0)
@mock.patch("shutil.copy")
def test_copy_project_config_no_overwrite(copy, _compute_crc32, exists):
    annif.hfh_util.copy_project_config(
        os.path.join("tests", "huggingface-cache", "dummy-fi.cfg"), force=False
    )
    assert not copy.called


@mock.patch("os.path.exists", return_value=True)
@mock.patch("shutil.copy")
def test_copy_project_config_overwrite(copy, exists):
    annif.hfh_util.copy_project_config(
        os.path.join("tests", "huggingface-cache", "dummy-fi.cfg"), force=True
    )
    assert copy.called
    assert copy.call_args == mock.call(
        "tests/huggingface-cache/dummy-fi.cfg", "projects.d/dummy-fi.cfg"
    )
