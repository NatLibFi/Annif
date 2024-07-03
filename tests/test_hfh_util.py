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


@mock.patch("annif.hfh_util._list_files_in_hf_hub", return_value=["README.md"])
@mock.patch(
    "huggingface_hub.ModelCard",
)
def test_upsert_modelcard_existing_card(ModelCard, _list_files_in_hf_hub, project):
    repo_id = "annif-user/Annif-HFH-repo"
    project.vocab_lang = "fi"
    projects = [project]
    token = "mytoken"
    revision = "main"
    ModelCard.load.return_value.data.language = ["en"]  # Mock language in card

    annif.hfh_util.upsert_modelcard(repo_id, projects, token, revision)

    ModelCard.assert_not_called()  # Do not create a new card

    ModelCard.load.assert_called_once_with(repo_id)
    card = ModelCard.load.return_value
    card.push_to_hub.assert_called_once_with(
        repo_id=repo_id,
        token=token,
        revision=revision,
        commit_message="Update README.md with Annif",
    )
    assert sorted(card.data.language) == ["en", "fi"]


@mock.patch("annif.hfh_util._list_files_in_hf_hub", return_value=[])
@mock.patch(
    "huggingface_hub.ModelCard",
)
def test_upsert_modelcard_new_card(ModelCard, _list_files_in_hf_hub, project):
    repo_id = "annif-user/Annif-HFH-repo"
    project.vocab_lang = "fi"
    projects = [project]
    token = "mytoken"
    revision = "main"

    annif.hfh_util.upsert_modelcard(repo_id, projects, token, revision)

    ModelCard.assert_called_once()
    card = ModelCard.return_value
    card.push_to_hub.assert_called_once_with(
        repo_id=repo_id,
        token=token,
        revision=revision,
        commit_message="Create README.md with Annif",
    )
    assert card.data.language == ["fi"]


@mock.patch(
    "huggingface_hub.ModelCard",
)
def test_create_modelcard(ModelCard):
    repo_id = "annif-user/Annif-HFH-repo"

    card = annif.hfh_util._create_modelcard(repo_id)

    assert "# Annif-HFH-repo" in ModelCard.call_args[0][0]  # README heading
    assert card.data.pipeline_tag == "text-classification"
    assert card.data.tags == ["annif"]
