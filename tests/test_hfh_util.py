"""Unit test module for Hugging Face Hub utilities."""

import io
import os.path
import zipfile
from datetime import datetime, timezone
from unittest import mock

import huggingface_hub
from huggingface_hub.utils import EntryNotFoundError

import annif.hfh_util
from annif.config import AnnifConfigCFG


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


@mock.patch(
    "huggingface_hub.ModelCard.load",
    side_effect=EntryNotFoundError("mymessage"),
)
@mock.patch("huggingface_hub.HfFileSystem.glob", return_value=[])
@mock.patch("huggingface_hub.ModelCard")
def test_upsert_modelcard_insert_new(ModelCard, glob, load, project):
    repo_id = "annif-user/annif-repo"
    token = "mytoken"
    revision = "mybranch"

    annif.hfh_util.upsert_modelcard(repo_id, [project], token, revision)

    ModelCard.assert_called_once()
    assert "# annif-repo" in ModelCard.call_args[0][0]  # README heading

    card = ModelCard.return_value
    assert card.data.language == ["fi"]
    assert card.data.pipeline_tag == "text-classification"
    assert card.data.tags == ["annif"]
    card.push_to_hub.assert_called_once_with(
        repo_id=repo_id,
        token=token,
        revision=revision,
        commit_message="Create README.md with Annif",
    )


@mock.patch("huggingface_hub.ModelCard.push_to_hub")
@mock.patch(
    "huggingface_hub.ModelCard.load",  # Mock language in existing card
    return_value=huggingface_hub.ModelCard("---\nlanguage:\n- en\n---"),
)
@mock.patch("huggingface_hub.HfFileSystem.glob", return_value=["dummy-en.cfg"])
@mock.patch(
    "huggingface_hub.HfFileSystem.read_text",
    return_value="""
        [dummy-en]
        name=Dummy English
        language=en
        vocab=dummy
""",
)
def test_upsert_modelcard_update_existing(read_text, glob, load, push_to_hub, project):
    repo_id = "annif-user/annif-repo"
    token = "mytoken"
    revision = "mybranch"

    annif.hfh_util.upsert_modelcard(repo_id, [project], token, revision)

    load.assert_called_once_with(repo_id)

    card = load.return_value
    retained_project_list_content = (
        "dummy-en            Dummy English           dummy           en"
    )
    assert retained_project_list_content in card.text
    assert sorted(card.data.language) == ["en", "fi"]
    card.push_to_hub.assert_called_once_with(
        repo_id=repo_id,
        token=token,
        revision=revision,
        commit_message="Update README.md with Annif",
    )


def test_update_modelcard_projects_section_append_new():
    empty_cfg = AnnifConfigCFG(projstr="")

    text = """This is some existing text in the card."""
    updated_text = annif.hfh_util._update_projects_section(text, empty_cfg)

    expected_tail = """\
<!--- start-of-autoupdating-part --->
## Projects
```
Project ID          Project Name            Vocabulary ID   Language
--------------------------------------------------------------------
```
<!--- end-of-autoupdating-part --->"""

    assert updated_text == text + expected_tail


def test_update_modelcard_projects_section_update_existing():
    cfg = AnnifConfigCFG(
        projstr="""\
        [dummy-fi]
        name=Dummy Finnish
        language=fi
        vocab=dummy"""
    )

    text_head = """This is some existing text in the card.\n"""

    text_initial_projects = """\
<!--- start-of-autoupdating-part --->
## Projects
```
Project ID          Project Name            Vocabulary ID   Language
--------------------------------------------------------------------
```
<!--- end-of-autoupdating-part --->\n"""

    text_tail = (
        "This is text after the Projects section; it should remain after updates."
    )

    text = text_head + text_initial_projects + text_tail
    updated_text = annif.hfh_util._update_projects_section(text, cfg)

    expected_updated_projects = """\
<!--- start-of-autoupdating-part --->
## Projects
```
Project ID          Project Name            Vocabulary ID   Language
--------------------------------------------------------------------
dummy-fi            Dummy Finnish           dummy           fi      \n```
<!--- end-of-autoupdating-part --->
"""

    assert updated_text == text_head + expected_updated_projects + text_tail
